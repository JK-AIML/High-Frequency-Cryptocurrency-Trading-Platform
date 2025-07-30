"""
Advanced alerting system for Tick Data Analysis & Alpha Detection.
"""

from typing import Dict, List, Optional, Callable
import time
from datetime import datetime
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
import requests
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: Callable[[Dict], bool]
    severity: AlertSeverity
    message_template: str
    cooldown: int = 300  # 5 minutes

class AlertManager:
    """Advanced alerting system."""
    
    def __init__(self, config: Dict):
        """Initialize alert manager.
        
        Args:
            config: Alert configuration
        """
        self.config = config
        self.rules: List[AlertRule] = []
        self.last_alert_time: Dict[str, float] = {}
        
        # Initialize notification channels
        self._setup_email()
        self._setup_slack()
        
    def _setup_email(self) -> None:
        """Setup email notification."""
        self.email_config = self.config.get('email', {})
        if self.email_config:
            self.smtp_server = smtplib.SMTP(
                self.email_config['smtp_server'],
                self.email_config['smtp_port']
            )
            self.smtp_server.starttls()
            self.smtp_server.login(
                self.email_config['username'],
                self.email_config['password']
            )
            
    def _setup_slack(self) -> None:
        """Setup Slack notification."""
        self.slack_config = self.config.get('slack', {})
        self.slack_webhook = self.slack_config.get('webhook_url')
        
    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule.
        
        Args:
            rule: Alert rule
        """
        self.rules.append(rule)
        
    def check_alerts(self, data: Dict) -> None:
        """Check alert rules against data.
        
        Args:
            data: Current system state
        """
        for rule in self.rules:
            try:
                # Check cooldown
                last_alert = self.last_alert_time.get(rule.name, 0)
                if time.time() - last_alert < rule.cooldown:
                    continue
                    
                # Check condition
                if rule.condition(data):
                    # Format message
                    message = rule.message_template.format(**data)
                    
                    # Send notifications
                    self._send_notifications(rule.severity, message)
                    
                    # Update last alert time
                    self.last_alert_time[rule.name] = time.time()
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule.name}: {e}")
                
    def _send_notifications(self, severity: AlertSeverity, message: str) -> None:
        """Send notifications through all channels.
        
        Args:
            severity: Alert severity
            message: Alert message
        """
        # Send email
        if self.email_config:
            self._send_email(severity, message)
            
        # Send Slack
        if self.slack_webhook:
            self._send_slack(severity, message)
            
    def _send_email(self, severity: AlertSeverity, message: str) -> None:
        """Send email notification.
        
        Args:
            severity: Alert severity
            message: Alert message
        """
        try:
            msg = MIMEText(message)
            msg['Subject'] = f"[{severity.value}] Trading System Alert"
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            
            self.smtp_server.send_message(msg)
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            
    def _send_slack(self, severity: AlertSeverity, message: str) -> None:
        """Send Slack notification.
        
        Args:
            severity: Alert severity
            message: Alert message
        """
        try:
            payload = {
                'text': f"*[{severity.value}]* Trading System Alert\n{message}"
            }
            
            requests.post(self.slack_webhook, json=payload)
            
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            
    def get_alert_history(self) -> List[Dict]:
        """Get alert history.
        
        Returns:
            List of alert history entries
        """
        return [
            {
                'rule': rule.name,
                'last_alert': datetime.fromtimestamp(time).isoformat(),
                'severity': rule.severity.value
            }
            for rule in self.rules
            for time in [self.last_alert_time.get(rule.name, 0)]
            if time > 0
        ] 