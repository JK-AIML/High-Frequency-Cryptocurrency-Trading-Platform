"""
Cloud Deployment Tests

Tests for deploying and validating the application on different cloud providers.
"""

import os
import unittest
import time
import subprocess
import json
from typing import Dict, Any, Optional
import boto3
from google.cloud import container_v1
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservice import ContainerServiceClient

class CloudDeploymentTest(unittest.TestCase):
    """Base class for cloud deployment tests."""
    
    PROVIDER = None  # To be set by subclasses
    CLUSTER_NAME = "tick-analysis-cluster"
    NAMESPACE = "tick-analysis"
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_id = f"test-{int(time.time())}"
        cls.kubeconfig_path = f"/tmp/kubeconfig-{cls.test_id}.yaml"
        
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.kubeconfig_path):
            os.remove(self.kubeconfig_path)
    
    def test_cluster_creation(self):
        """Test cluster creation on the target cloud provider."""
        self.assertTrue(self._create_cluster(), "Cluster creation failed")
    
    def test_application_deployment(self):
        """Test deploying the application to the cluster."""
        self.assertTrue(self._deploy_application(), "Application deployment failed")
    
    def test_application_scaling(self):
        """Test application scaling functionality."""
        if not self._deploy_application():
            self.skipTest("Application deployment failed")
            
        # Test horizontal scaling
        self._test_horizontal_scaling()
        
        # Test vertical scaling
        self._test_vertical_scaling()
    
    def test_load_testing(self):
        """Run load tests against the deployed application."""
        if not self._deploy_application():
            self.skipTest("Application deployment failed")
            
        self._run_load_tests()
    
    def _create_cluster(self) -> bool:
        """Create a Kubernetes cluster on the target cloud provider."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def _deploy_application(self) -> bool:
        """Deploy the application to the cluster."""
        try:
            # Apply namespace
            self._run_kubectl("create", "namespace", self.NAMESPACE, can_fail=True)
            
            # Apply all resources
            kustomize_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "deploy", "kubernetes"
            )
            
            # Apply base resources
            self._run_kubectl(
                "apply", "-k", 
                os.path.join(kustomize_dir, "base"),
                "-n", self.NAMESPACE
            )
            
            # Wait for deployments to be ready
            self._wait_for_deployments()
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Deployment failed: {e}")
            return False
    
    def _test_horizontal_scaling(self):
        """Test horizontal pod autoscaling."""
        # Deploy HPA
        hpa_manifest = """
        apiVersion: autoscaling/v2
        kind: HorizontalPodAutoscaler
        metadata:
          name: tick-analysis-scaler
        spec:
          scaleTargetRef:
            apiVersion: apps/v1
            kind: Deployment
            name: tick-analysis
          minReplicas: 1
          maxReplicas: 10
          metrics:
          - type: Resource
            resource:
              name: cpu
              target:
                type: Utilization
                averageUtilization: 80
        """
        
        self._run_kubectl("apply", "-f", "-", "-n", self.NAMESPACE, input_data=hpa_manifest)
        
        # Run load test to trigger scaling
        self._run_load_tests()
        
        # Verify scaling occurred
        time.sleep(60)  # Wait for metrics to propagate
        hpa = json.loads(self._run_kubectl("get", "hpa", "tick-analysis-scaler", "-o", "json", "-n", self.NAMESPACE))
        self.assertGreater(hpa["status"]["currentReplicas"], 1, "HPA did not scale out")
    
    def _test_vertical_scaling(self):
        """Test vertical pod autoscaling."""
        # Deploy VPA
        vpa_manifest = """
        apiVersion: autoscaling.k8s.io/v1
        kind: VerticalPodAutoscaler
        metadata:
          name: tick-analysis-vpa
        spec:
          targetRef:
            apiVersion: "apps/v1"
            kind:       Deployment
            name:       tick-analysis
          updatePolicy:
            updateMode: "Auto"
        """
        
        self._run_kubectl("apply", "-f", "-", "-n", self.NAMESPACE, input_data=vpa_manifest)
        
        # Run load test to trigger scaling
        self._run_load_tests()
        
        # Verify VPA recommendations
        time.sleep(60)  # Wait for metrics to propagate
        vpa = json.loads(self._run_kubectl("get", "vpa", "tick-analysis-vpa", "-o", "json", "-n", self.NAMESPACE))
        self.assertIn("recommendation", vpa["status"], "No VPA recommendations available")
    
    def _run_load_tests(self):
        """Run load tests against the deployed application."""
        # Get service URL
        service_url = self._get_service_url()
        
        # Run locust load test
        locust_cmd = [
            "locust",
            "-f", "tests/load_test.py",
            "--host", service_url,
            "--users", "100",
            "--spawn-rate", "10",
            "--run-time", "5m",
            "--headless"
        ]
        
        try:
            subprocess.run(locust_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Load test failed: {e.stderr}")
            raise
    
    def _get_service_url(self) -> str:
        """Get the URL of the deployed service."""
        # For cloud providers, this would typically be a load balancer or ingress
        # For testing, we'll use port-forwarding
        port = 8080
        
        # Start port-forward in background
        self.port_forward = subprocess.Popen(
            ["kubectl", "port-forward", "svc/tick-analysis", f"{port}:80", "-n", self.NAMESPACE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for port-forward to be ready
        time.sleep(5)
        return f"http://localhost:{port}"
    
    def _wait_for_deployments(self, timeout: int = 300):
        """Wait for all deployments to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployments = json.loads(
                    self._run_kubectl("get", "deployments", "-o", "json", "-n", self.NAMESPACE)
                )
                
                ready = True
                for deploy in deployments.get("items", []):
                    for condition in deploy.get("status", {}).get("conditions", []):
                        if condition.get("type") == "Available" and condition.get("status") != "True":
                            ready = False
                            break
                    
                    if not ready:
                        break
                
                if ready:
                    return
                
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                pass
                
            time.sleep(5)
        
        raise TimeoutError("Timed out waiting for deployments to be ready")
    
    def _run_kubectl(self, *args, input_data: str = None, can_fail: bool = False) -> str:
        """Run a kubectl command."""
        cmd = ["kubectl", "--kubeconfig", self.kubeconfig_path] + list(args)
        
        try:
            result = subprocess.run(
                cmd,
                input=input_data,
                capture_output=True,
                text=True,
                check=not can_fail
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            if not can_fail:
                print(f"Command failed: {e.cmd}")
                print(f"Error: {e.stderr}")
                raise
            return ""


class AWSDeploymentTest(CloudDeploymentTest):
    """AWS EKS deployment tests."""
    
    PROVIDER = "aws"
    
    def _create_cluster(self) -> bool:
        """Create an EKS cluster."""
        try:
            # Create EKS cluster using AWS CLI or boto3
            # This is a simplified example - in practice, you'd want to use Terraform or CloudFormation
            cluster_config = f"""
            apiVersion: eksctl.io/v1alpha5
            kind: ClusterConfig
            metadata:
              name: {self.CLUSTER_NAME}
              region: us-west-2
            nodeGroups:
              - name: ng-1
                desiredCapacity: 2
                minSize: 1
                maxSize: 5
                instanceType: t3.medium
            """
            
            with open("eks-cluster.yaml", "w") as f:
                f.write(cluster_config)
            
            subprocess.run(
                ["eksctl", "create", "cluster", "-f", "eks-cluster.yaml"],
                check=True
            )
            
            # Get kubeconfig
            subprocess.run(
                ["aws", "eks", "update-kubeconfig", 
                 "--name", self.CLUSTER_NAME,
                 "--kubeconfig", self.kubeconfig_path],
                check=True
            )
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to create EKS cluster: {e}")
            return False


class GCPDeploymentTest(CloudDeploymentTest):
    """GCP GKE deployment tests."""
    
    PROVIDER = "gcp"
    
    def _create_cluster(self) -> bool:
        """Create a GKE cluster."""
        try:
            # Create GKE cluster using gcloud CLI
            subprocess.run(
                ["gcloud", "container", "clusters", "create", self.CLUSTER_NAME,
                 "--num-nodes=2", "--machine-type=n1-standard-2", "--region=us-central1"],
                check=True
            )
            
            # Get kubeconfig
            subprocess.run(
                ["gcloud", "container", "clusters", "get-credentials", 
                 self.CLUSTER_NAME, "--region=us-central1",
                 "--kubeconfig", self.kubeconfig_path],
                check=True
            )
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to create GKE cluster: {e}")
            return False


class AzureDeploymentTest(CloudDeploymentTest):
    """Azure AKS deployment tests."""
    
    PROVIDER = "azure"
    
    def _create_cluster(self) -> bool:
        """Create an AKS cluster."""
        try:
            # Create resource group
            subprocess.run(
                ["az", "group", "create", "--name", "tick-analysis-rg", "--location", "eastus"],
                check=True
            )
            
            # Create AKS cluster
            subprocess.run(
                ["az", "aks", "create",
                 "--resource-group", "tick-analysis-rg",
                 "--name", self.CLUSTER_NAME,
                 "--node-count", "2",
                 "--node-vm-size", "Standard_DS2_v2",
                 "--generate-ssh-keys"],
                check=True
            )
            
            # Get kubeconfig
            subprocess.run(
                ["az", "aks", "get-credentials",
                 "--resource-group", "tick-analysis-rg",
                 "--name", self.CLUSTER_NAME,
                 "--file", self.kubeconfig_path],
                check=True
            )
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to create AKS cluster: {e}")
            return False
