"""
Data Retention Policy Manager

This module provides functionality for managing data retention policies,
including automatic cleanup of old data based on configurable rules.
"""

import os
import shutil
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import re
import fnmatch

logger = logging.getLogger(__name__)

class RetentionGranularity(Enum):
    """Granularity levels for retention policies."""
    SECONDS = 'seconds'
    MINUTES = 'minutes'
    HOURS = 'hours'
    DAYS = 'days'
    WEEKS = 'weeks'
    MONTHS = 'months'
    YEARS = 'years'

@dataclass
class RetentionRule:
    """Defines a retention rule for data cleanup."""
    name: str
    pattern: str  # File pattern or table name pattern
    max_age: Optional[Union[int, timedelta]] = None  # Maximum age to keep
    max_count: Optional[int] = None  # Maximum number of items to keep
    min_free_space: Optional[float] = None  # Minimum free space in GB
    action: str = 'delete'  # 'delete', 'archive', 'compress', or custom action
    granularity: RetentionGranularity = RetentionGranularity.DAYS
    recursive: bool = True  # For file patterns
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert max_age to timedelta if it's an integer."""
        if isinstance(self.max_age, int):
            if self.granularity == RetentionGranularity.SECONDS:
                self.max_age = timedelta(seconds=self.max_age)
            elif self.granularity == RetentionGranularity.MINUTES:
                self.max_age = timedelta(minutes=self.max_age)
            elif self.granularity == RetentionGranularity.HOURS:
                self.max_age = timedelta(hours=self.max_age)
            elif self.granularity == RetentionGranularity.DAYS:
                self.max_age = timedelta(days=self.max_age)
            elif self.granularity == RetentionGranularity.WEEKS:
                self.max_age = timedelta(weeks=self.max_age)
            elif self.granularity == RetentionGranularity.MONTHS:
                # Approximate month as 30 days
                self.max_age = timedelta(days=30 * self.max_age)
            elif self.granularity == RetentionGranularity.YEARS:
                # Approximate year as 365 days
                self.max_age = timedelta(days=365 * self.max_age)

class RetentionAction(ABC):
    """Base class for retention actions."""
    
    @abstractmethod
    async def apply(self, target: Any, rule: RetentionRule) -> bool:
        """Apply the retention action to the target."""
        pass

class DeleteAction(RetentionAction):
    """Delete files or data that match the retention rule."""
    
    async def apply(self, target: str, rule: RetentionRule) -> bool:
        """Delete the target file or directory."""
        try:
            if os.path.isfile(target):
                os.remove(target)
                logger.info(f"Deleted file: {target}")
                return True
            elif os.path.isdir(target):
                if rule.recursive:
                    shutil.rmtree(target)
                    logger.info(f"Deleted directory (recursive): {target}")
                else:
                    os.rmdir(target)
                    logger.info(f"Deleted directory: {target}")
                return True
            else:
                logger.warning(f"Target does not exist: {target}")
                return False
        except Exception as e:
            logger.error(f"Error deleting {target}: {e}")
            return False

class ArchiveAction(RetentionAction):
    """Archive files before deletion."""
    
    def __init__(self, archive_dir: str, format: str = 'zip'):
        """
        Initialize the archive action.
        
        Args:
            archive_dir: Directory to store archives
            format: Archive format (zip, tar, gztar, bztar, xztar)
        """
        self.archive_dir = archive_dir
        self.format = format
        os.makedirs(archive_dir, exist_ok=True)
    
    async def apply(self, target: str, rule: RetentionRule) -> bool:
        """Archive the target before deletion."""
        if not os.path.exists(target):
            logger.warning(f"Target does not exist: {target}")
            return False
            
        try:
            # Create archive name based on target path and timestamp
            base_name = os.path.basename(target.rstrip('/\\'))
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            archive_name = f"{base_name}_{timestamp}"
            archive_path = os.path.join(self.archive_dir, archive_name)
            
            # Create the archive
            if os.path.isfile(target):
                shutil.make_archive(archive_path, self.format, os.path.dirname(target), base_name)
            else:  # Directory
                shutil.make_archive(archive_path, self.format, target)
            
            logger.info(f"Archived {target} to {archive_path}.{self.format}")
            
            # Now delete the original
            if os.path.isfile(target):
                os.remove(target)
            else:
                shutil.rmtree(target)
                
            return True
            
        except Exception as e:
            logger.error(f"Error archiving {target}: {e}")
            return False

class RetentionManager:
    """Manages data retention policies and cleanup operations."""
    
    def __init__(self, base_path: str = None):
        """
        Initialize the retention manager.
        
        Args:
            base_path: Base path for file operations (if None, uses current directory)
        """
        self.base_path = os.path.abspath(base_path or os.getcwd())
        self.rules: Dict[str, RetentionRule] = {}
        self.actions = {
            'delete': DeleteAction(),
            'archive': ArchiveAction(os.path.join(os.getcwd(), 'archives'))
        }
        self._running = False
        self._cleanup_task = None
    
    def add_rule(self, rule: RetentionRule) -> None:
        """Add a retention rule."""
        self.rules[rule.name] = rule
    
    def remove_rule(self, name: str) -> bool:
        """Remove a retention rule by name."""
        if name in self.rules:
            del self.rules[name]
            return True
        return False
    
    def add_action(self, name: str, action: RetentionAction) -> None:
        """Add a custom retention action."""
        self.actions[name] = action
    
    async def start(self, interval: int = 3600) -> None:
        """
        Start the retention manager with periodic cleanup.
        
        Args:
            interval: Seconds between cleanup runs
        """
        if self._running:
            logger.warning("Retention manager is already running")
            return
            
        self._running = True
        
        async def run_cleanup():
            while self._running:
                try:
                    await self.cleanup()
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")
                
                # Wait for the next interval, but check running flag frequently
                for _ in range(interval):
                    if not self._running:
                        break
                    await asyncio.sleep(1)
        
        self._cleanup_task = asyncio.create_task(run_cleanup())
        logger.info(f"Started retention manager with {len(self.rules)} rules")
    
    async def stop(self) -> None:
        """Stop the retention manager."""
        self._running = False
        if self._cleanup_task:
            await self._cleanup_task
            self._cleanup_task = None
        logger.info("Stopped retention manager")
    
    async def cleanup(self) -> Dict[str, Dict[str, Any]]:
        """
        Run cleanup based on all retention rules.
        
        Returns:
            Dictionary of cleanup results by rule name
        """
        results = {}
        
        for name, rule in self.rules.items():
            try:
                results[name] = await self.apply_rule(rule)
            except Exception as e:
                logger.error(f"Error applying rule '{name}': {e}")
                results[name] = {
                    'success': False,
                    'error': str(e),
                    'processed': 0,
                    'deleted': 0,
                    'skipped': 0,
                    'failed': 0
                }
        
        return results
    
    async def apply_rule(self, rule: Union[str, RetentionRule]) -> Dict[str, Any]:
        """
        Apply a single retention rule.
        
        Args:
            rule: Either a RetentionRule object or rule name
            
        Returns:
            Dictionary with cleanup results
        """
        if isinstance(rule, str):
            if rule not in self.rules:
                raise ValueError(f"No such rule: {rule}")
            rule = self.rules[rule]
        
        logger.info(f"Applying retention rule: {rule.name}")
        
        # Get the appropriate action
        action = self.actions.get(rule.action)
        if not action:
            raise ValueError(f"No such action: {rule.action}")
        
        # Find matching files/directories
        matches = await self._find_matches(rule)
        
        # Apply retention criteria
        to_process = []
        now = datetime.now()
        
        for path, stats in matches:
            # Check max age
            if rule.max_age is not None:
                file_time = datetime.fromtimestamp(stats.st_mtime)
                if (now - file_time) < rule.max_age:
                    continue
            
            to_process.append((path, stats))
        
        # Apply max count (keep most recent N)
        if rule.max_count is not None and len(to_process) > rule.max_count:
            # Sort by modification time (newest first) and keep max_count
            to_process.sort(key=lambda x: x[1].st_mtime, reverse=True)
            to_process = to_process[rule.max_count:]
        
        # Check free space if needed
        if rule.min_free_space is not None:
            # Get disk usage
            total, used, free = shutil.disk_usage(os.path.dirname(rule.pattern) if rule.pattern != '/' else '/')
            free_gb = free / (2**30)  # Convert to GB
            
            if free_gb > rule.min_free_space:
                # Enough free space, no need to delete anything
                return {
                    'success': True,
                    'processed': 0,
                    'deleted': 0,
                    'skipped': len(to_process),
                    'failed': 0,
                    'free_space_gb': free_gb
                }
        
        # Apply the action to each item
        results = {
            'processed': 0,
            'deleted': 0,
            'skipped': 0,
            'failed': 0
        }
        
        for path, _ in to_process:
            try:
                success = await action.apply(path, rule)
                if success:
                    results['deleted'] += 1
                else:
                    results['failed'] += 1
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results['failed'] += 1
            
            results['processed'] += 1
        
        results['success'] = results['failed'] == 0
        return results
    
    async def _find_matches(self, rule: RetentionRule) -> List[tuple]:
        """Find files/directories matching the rule pattern."""
        import glob
        import os
        import fnmatch
        
        # Handle absolute and relative paths
        if os.path.isabs(rule.pattern):
            search_path = rule.pattern
        else:
            search_path = os.path.join(self.base_path, rule.pattern)
        
        # Check if it's a direct file/directory match
        if os.path.exists(search_path):
            return [(search_path, os.stat(search_path))]
        
        # Handle glob patterns
        matches = []
        for root, dirs, files in os.walk(os.path.dirname(search_path) or '.', topdown=True):
            # Filter directories if not recursive
            if not rule.recursive:
                dirs[:] = []
            
            # Check directory against pattern
            dir_path = os.path.join(root, '')
            if fnmatch.fnmatch(dir_path, search_path):
                try:
                    matches.append((dir_path, os.stat(dir_path)))
                except OSError:
                    continue
            
            # Check files against pattern
            for file in files:
                file_path = os.path.join(root, file)
                if fnmatch.fnmatch(file_path, search_path):
                    try:
                        matches.append((file_path, os.stat(file_path)))
                    except OSError:
                        continue
        
        return matches

# Example usage
if __name__ == "__main__":
    import asyncio
    import tempfile
    import time
    
    async def example():
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")
            
            # Create some test files
            for i in range(5):
                file_path = os.path.join(temp_dir, f"test_{i}.txt")
                with open(file_path, 'w') as f:
                    f.write(f"Test file {i}")
                # Set file modification time to i days ago
                days_ago = time.time() - (i * 86400)
                os.utime(file_path, (days_ago, days_ago))
            
            # Set up retention manager
            manager = RetentionManager(temp_dir)
            
            # Add a rule to delete files older than 2 days, keeping at least 2 files
            rule = RetentionRule(
                name="cleanup_old_files",
                pattern="*.txt",
                max_age=2,  # 2 days
                max_count=2,  # Keep at least 2 most recent files
                action="delete"
            )
            manager.add_rule(rule)
            
            # Run cleanup
            print("Running cleanup...")
            results = await manager.cleanup()
            
            # Print results
            for rule_name, result in results.items():
                print(f"\nResults for {rule_name}:")
                print(f"  Success: {result.get('success', False)}")
                print(f"  Processed: {result.get('processed', 0)}")
                print(f"  Deleted: {result.get('deleted', 0)}")
                print(f"  Skipped: {result.get('skipped', 0)}")
                print(f"  Failed: {result.get('failed', 0)}")
            
            # List remaining files
            print("\nRemaining files:")
            for file in os.listdir(temp_dir):
                print(f"  - {file}")
    
    # Run the example
    asyncio.run(example())
