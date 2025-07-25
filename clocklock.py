#!/usr/bin/env python3
"""
ClockLock - Optimized Time Lock Tool
A secure file/folder locking application with time-based unlocking mechanism.
"""

import json
import requests
import os
import sys
import zipfile
import threading
import time
import math
import base64
import hashlib
import hmac
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import timezone, timedelta, datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache, wraps
from dataclasses import dataclass, asdict
import logging

# GUI imports
import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkcalendar import DateEntry
from email.utils import parsedate_to_datetime
from cryptography.fernet import Fernet

# Optional imports with fallbacks
try:
    import pygame
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False


# Constants
CONFIG_FILE = "config.json"
HISTORY_FILE = "unlock_history.json"
RECENT_FILES_FILE = "recent_files.json"
SECRET_KEY = b"lockthisshituwontseeitagain"
MAX_HISTORY_ENTRIES = 50
MAX_RECENT_FILES = 20
CHUNK_SIZE = 8192  # For file I/O operations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clocklock.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize pygame mixer for sound notifications
if SOUND_AVAILABLE:
    try:
        pygame.mixer.init()
    except Exception as e:
        logger.warning(f"Failed to initialize pygame mixer: {e}")
        SOUND_AVAILABLE = False

@dataclass
class FileMetadata:
    """Structured metadata for locked files."""
    original_filename: str
    unlock_utc: str
    timezone_offset: int
    size: int
    encryption_key: str
    signature: str = ""
    is_folder: bool = False
    is_batch: bool = False
    original_deleted: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileMetadata':
        """Create instance from dictionary."""
        return cls(**data)


class ConfigManager:
    """Manages application configuration with validation and caching."""
    
    _instance = None
    _config_cache = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def default_config(self) -> Dict[str, Any]:
        """Default configuration values."""
        return {
            "timezone_offset": 1,
            "default_output_folder": "",
            "dark_mode": 1,
            "timeout_seconds": 3,
            "max_filename_length": 40,
            "max_file_size_mb": 500,
            "sound_notifications": True,
            "show_file_icons": True,
            "auto_relock": False,
            "delete_original_files": True,
            "selected_timezone": "UTC+1",
            "quick_presets": {
                "1 Hour": 1,
                "1 Day": 24,
                "1 Week": 168,
                "1 Month": 720
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load and validate configuration."""
        if self._config_cache is not None:
            return self._config_cache
        
        try:
            if not Path(CONFIG_FILE).exists():
                self._create_default_config()
            
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Merge with defaults for missing keys
            updated = False
            for key, value in self.default_config.items():
                if key not in config_data:
                    config_data[key] = value
                    updated = True
            
            if updated:
                self.save_config(config_data)
            
            self._config_cache = config_data
            logger.info("Configuration loaded successfully")
            return config_data
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self.default_config
    
    def save_config(self, config_data: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4)
            self._config_cache = config_data.copy()
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _create_default_config(self) -> None:
        """Create default configuration file."""
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.default_config, f, indent=4)
        logger.info("Created default config.json")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        config = self.load_config()
        return config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        config = self.load_config()
        config[key] = value
        self.save_config(config)


class FileManager:
    """Handles file operations with optimized I/O and caching."""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def get_file_icon(filepath: str) -> str:
        """Return appropriate emoji for file type with caching."""
        if os.path.isdir(filepath):
            return "📁"
        
        ext = Path(filepath).suffix.lower()
        
        icon_map = {
            '.txt': '📄', '.doc': '📄', '.docx': '📄', '.pdf': '📄',
            '.jpg': '🖼️', '.jpeg': '🖼️', '.png': '🖼️', '.gif': '🖼️', '.bmp': '🖼️',
            '.mp4': '🎬', '.avi': '🎬', '.mov': '🎬', '.mkv': '🎬',
            '.mp3': '🎵', '.wav': '🎵', '.flac': '🎵', '.ogg': '🎵',
            '.zip': '📦', '.rar': '📦', '.7z': '📦', '.tar': '📦',
            '.exe': '⚙️', '.msi': '⚙️', '.deb': '⚙️', '.dmg': '⚙️',
            '.py': '🐍', '.js': '📜', '.html': '🌐', '.css': '🎨',
            '.xlsx': '📊', '.csv': '📊', '.db': '🗃️', '.json': '⚙️'
        }
        
        return icon_map.get(ext, '📄')
    
    @staticmethod
    def get_file_size_mb(filepath: str) -> float:
        """Get file size in MB with error handling."""
        try:
            return Path(filepath).stat().st_size / (1024 * 1024)
        except (OSError, FileNotFoundError) as e:
            logger.warning(f"Could not get size for {filepath}: {e}")
            return 0.0
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"
    
    @staticmethod
    def shorten_filename(name: str, max_len: int = 40) -> str:
        """Shorten long filenames for display."""
        return name if len(name) <= max_len else f"{name[:max_len - 3]}..."
    
    @staticmethod
    def safe_file_operation(operation, *args, **kwargs):
        """Wrapper for safe file operations with retry mechanism."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except (OSError, IOError) as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
        

class CryptoManager:
    """Handles encryption/decryption operations with security focus."""
    
    @staticmethod
    def create_hmac(metadata_dict: Dict[str, Any], key: bytes = SECRET_KEY) -> str:
        """Create HMAC signature for metadata."""
        # Remove signature from dict for calculation
        clean_dict = {k: v for k, v in metadata_dict.items() if k != 'signature'}
        meta_json = json.dumps(clean_dict, sort_keys=True).encode('utf-8')
        return hmac.new(key, meta_json, hashlib.sha256).hexdigest()
    
    @staticmethod
    def verify_hmac(metadata_dict: Dict[str, Any], signature: str, key: bytes = SECRET_KEY) -> bool:
        """Verify HMAC signature for metadata integrity."""
        try:
            expected = CryptoManager.create_hmac(metadata_dict, key)
            return hmac.compare_digest(expected, signature)
        except Exception as e:
            logger.error(f"HMAC verification failed: {e}")
            return False
    
    @staticmethod
    def encrypt_data(data: bytes, progress_callback=None) -> Tuple[bytes, bytes]:
        """Encrypt data and return encrypted data and key."""
        key = Fernet.generate_key()
        fernet = Fernet(key)
        
        if progress_callback:
            progress_callback(0.5, "Encrypting data...")
        
        encrypted = fernet.encrypt(data)
        
        if progress_callback:
            progress_callback(1.0, "Encryption complete")
        
        return encrypted, key
    
    @staticmethod
    def decrypt_data(encrypted_data: bytes, key: bytes, progress_callback=None) -> bytes:
        """Decrypt data using provided key."""
        fernet = Fernet(key)
        
        if progress_callback:
            progress_callback(0.5, "Decrypting data...")
        
        decrypted = fernet.decrypt(encrypted_data)
        
        if progress_callback:
            progress_callback(1.0, "Decryption complete")
        
        return decrypted


class HistoryManager:
    """Manages unlock history and recent files."""
    
    @staticmethod
    def load_json_file(filepath: str, default: List = None) -> List[Dict[str, Any]]:
        """Load JSON file with error handling."""
        if default is None:
            default = []
        
        try:
            if Path(filepath).exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
        
        return default
    
    @staticmethod
    def save_json_file(filepath: str, data: List[Dict[str, Any]]) -> None:
        """Save JSON file with error handling."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")
    
    @classmethod
    def load_history(cls) -> List[Dict[str, Any]]:
        """Load unlock history."""
        return cls.load_json_file(HISTORY_FILE)
    
    @classmethod
    def save_history(cls, entry: Dict[str, Any]) -> None:
        """Save entry to unlock history."""
        history = cls.load_history()
        history.insert(0, entry)
        history = history[:MAX_HISTORY_ENTRIES]  # Keep only recent entries
        cls.save_json_file(HISTORY_FILE, history)
    
    @classmethod
    def load_recent_files(cls) -> List[Dict[str, Any]]:
        """Load recent files list."""
        return cls.load_json_file(RECENT_FILES_FILE)
    
    @classmethod
    def save_recent_file(cls, filepath: str, action: str) -> None:
        """Save entry to recent files."""
        recent = cls.load_recent_files()
        entry = {
            "path": filepath,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "filename": Path(filepath).name
        }
        
        # Remove duplicates
        recent = [r for r in recent if r["path"] != filepath]
        recent.insert(0, entry)
        recent = recent[:MAX_RECENT_FILES]  # Keep only recent entries
        
        cls.save_json_file(RECENT_FILES_FILE, recent)


class TimeManager:
    """Handles time operations with caching and fallbacks."""
    
    @staticmethod
    @lru_cache(maxsize=1)
    def _get_cached_time(cache_key: int) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get internet time with short-term caching (1 minute cache)."""
        return TimeManager._fetch_internet_time()
    
    @staticmethod
    def _fetch_internet_time() -> Tuple[Optional[datetime], Optional[datetime]]:
        """Fetch internet time from multiple sources."""
        urls = ["https://www.cloudflare.com", "https://www.google.com", "https://httpbin.org/delay/0"]
        config_manager = ConfigManager()
        timeout = config_manager.get("timeout_seconds", 3)
        timezone_offset = config_manager.get("timezone_offset", 1)
        
        for url in urls:
            try:
                response = requests.get(url, timeout=timeout)
                date_str = response.headers.get("Date")
                if date_str:
                    utc_time = parsedate_to_datetime(date_str)
                    local_time = utc_time.astimezone(timezone(timedelta(hours=timezone_offset)))
                    return utc_time, local_time
            except Exception as e:
                logger.warning(f"Failed to get time from {url}: {e}")
        
        # Fallback to system time
        logger.warning("Using system time as fallback")
        utc_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        local_time = utc_time.astimezone(timezone(timedelta(hours=timezone_offset)))
        return utc_time, local_time
    
    @classmethod
    def get_internet_date(cls, timezone_offset: int = 0, timeout_seconds: int = 3) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get internet date with caching (cache expires every minute)."""
        cache_key = int(time.time() // 60)  # Cache for 1 minute
        try:
            # Clear cache if it's too old
            cls._get_cached_time.cache_clear()
            return cls._get_cached_time(cache_key)
        except Exception as e:
            logger.error(f"Failed to get internet time: {e}")
            return None, None
    
    @staticmethod
    def format_time_remaining(seconds: float) -> str:
        """Format time remaining in HH:MM:SS format."""
        if seconds <= 0:
            return "00:00:00"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"


class SoundManager:
    """Handles sound notifications with fallbacks."""
    
    @staticmethod
    def play_notification_sound() -> None:
        """Play notification sound if available and enabled."""
        config_manager = ConfigManager()
        
        if not SOUND_AVAILABLE or not config_manager.get("sound_notifications", True):
            return
        
        try:
            # Create a simple beep sound
            duration = 0.5
            sample_rate = 22050
            frames = int(duration * sample_rate)
            arr = []
            
            for i in range(frames):
                wave = 4096 * math.sin(2 * math.pi * 800 * i / sample_rate)
                arr.append([int(wave), int(wave)])
            
            sound = pygame.sndarray.make_sound(arr)
            sound.play()
            
        except Exception as e:
            logger.warning(f"Failed to play notification sound: {e}")


def handle_exceptions(func):
    """Decorator for exception handling with user-friendly error messages."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            return None
    return wrapper


class ProgressDialog:
    """Enhanced progress dialog with better UX."""
    
    def __init__(self, parent, title: str, message: str):
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("450x180")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self._center_window()
        
        # Create UI elements
        self._create_ui(message)
        
        # Track completion
        self.completed = False
    
    def _center_window(self) -> None:
        """Center dialog on screen."""
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (450 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (180 // 2)
        self.dialog.geometry(f"450x180+{x}+{y}")
    
    def _create_ui(self, message: str) -> None:
        """Create UI elements."""
        self.label = ctk.CTkLabel(
            self.dialog, 
            text=message, 
            font=ctk.CTkFont(size=14),
            wraplength=400
        )
        self.label.pack(pady=20)
        
        self.progress = ctk.CTkProgressBar(self.dialog, width=350)
        self.progress.pack(pady=10)
        self.progress.set(0)
        
        self.percent_label = ctk.CTkLabel(self.dialog, text="0%")
        self.percent_label.pack(pady=5)
        
        # Add cancel button for long operations
        self.cancel_button = ctk.CTkButton(
            self.dialog, 
            text="Cancel", 
            command=self._on_cancel,
            width=100,
            height=30
        )
        self.cancel_button.pack(pady=10)
        
        self.cancelled = False
    
    def _on_cancel(self) -> None:
        """Handle cancel button click."""
        self.cancelled = True
        self.close()
    
    def update_progress(self, value: float, text: Optional[str] = None) -> None:
        """Update progress bar and text."""
        if self.cancelled:
            return
        
        try:
            self.progress.set(max(0, min(1, value)))
            self.percent_label.configure(text=f"{int(value * 100)}%")
            
            if text:
                self.label.configure(text=text)
            
            self.dialog.update()
            
            if value >= 1.0:
                self.completed = True
                self.cancel_button.configure(text="Close")
                
        except Exception as e:
            logger.warning(f"Progress dialog update failed: {e}")
    
    def close(self) -> None:
        """Close the dialog."""
        try:
            self.dialog.destroy()
        except Exception as e:
            logger.warning(f"Failed to close progress dialog: {e}")


class CountdownTimer:
    """Enhanced countdown timer with better performance."""
    
    def __init__(self, countdown_label, unlock_datetime: datetime, unlock_callback=None):
        self.countdown_label = countdown_label
        self.unlock_datetime = unlock_datetime
        self.unlock_callback = unlock_callback
        self.running = False
        self.thread = None
        self.sound_played = False
        self._stop_event = threading.Event()
    
    def start(self) -> None:
        """Start the countdown timer."""
        if not self.running:
            self.running = True
            self._stop_event.clear()
            self.thread = threading.Thread(target=self._countdown_loop, daemon=True)
            self.thread.start()
            logger.info("Countdown timer started")
    
    def stop(self) -> None:
        """Stop the countdown timer."""
        self.running = False
        self._stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Countdown timer stopped")
    
    def _countdown_loop(self) -> None:
        """Main countdown loop with optimized updates."""
        config_manager = ConfigManager()
        
        while self.running and not self._stop_event.is_set():
            try:
                utc_now, _ = TimeManager.get_internet_date(
                    config_manager.get("timezone_offset", 1),
                    config_manager.get("timeout_seconds", 3)
                )
                
                if utc_now:
                    time_remaining = (self.unlock_datetime - utc_now).total_seconds()
                    
                    if time_remaining <= 0:
                        self._handle_unlock_ready()
                        break
                    else:
                        self._update_countdown_display(time_remaining)
                
                # Sleep for 1 second or until stop event
                self._stop_event.wait(1.0)
                
            except Exception as e:
                logger.error(f"Countdown timer error: {e}")
                self._stop_event.wait(5.0)  # Wait longer on error
    
    def _handle_unlock_ready(self) -> None:
        """Handle when file is ready to unlock."""
        try:
            self.countdown_label.configure(
                text="✅ File can now be unlocked!",
                text_color="#00ff00"
            )
            
            # Play notification sound
            if not self.sound_played:
                SoundManager.play_notification_sound()
                self.sound_played = True
            
            if self.unlock_callback:
                self.unlock_callback()
            
            self.running = False
            
        except Exception as e:
            logger.error(f"Error handling unlock ready: {e}")
    
    def _update_countdown_display(self, time_remaining: float) -> None:
        """Update countdown display."""
        try:
            time_str = TimeManager.format_time_remaining(time_remaining)
            days = int(time_remaining // 86400)
            
            if days > 0:
                display_text = f"🔐 Unlocks in: {days}d {time_str}"
            else:
                display_text = f"🔐 Unlocks in: {time_str}"
            
            self.countdown_label.configure(
                text=display_text,
                text_color="#ffffff"
            )
            
        except Exception as e:
            logger.error(f"Error updating countdown display: {e}")


class FileEncryption:
    """Handles file encryption/decryption operations."""
    
    @staticmethod
    @handle_exceptions
    def encrypt_file(filepath: str, unlock_utc_time: datetime, 
                    is_folder: bool = False, delete_original: bool = True,
                    progress_callback=None) -> Optional[str]:
        """Encrypt a single file."""
        try:
            if progress_callback:
                progress_callback(0.1, "Reading file...")
            
            # Read file data in chunks for large files
            file_data = FileEncryption._read_file_chunked(filepath)
            
            if progress_callback:
                progress_callback(0.3, "Encrypting data...")
            
            # Encrypt data
            encrypted_data, encryption_key = CryptoManager.encrypt_data(
                file_data, 
                lambda p, t: progress_callback(0.3 + p * 0.4, t) if progress_callback else None
            )
            
            if progress_callback:
                progress_callback(0.7, "Writing encrypted file...")
            
            # Write encrypted file
            locked_path = filepath + ".locked"
            FileEncryption._write_file_chunked(locked_path, encrypted_data)
            
            if progress_callback:
                progress_callback(0.8, "Creating metadata...")
            
            # Create metadata
            metadata = FileMetadata(
                original_filename=Path(filepath).name,
                unlock_utc=unlock_utc_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                timezone_offset=ConfigManager().get("timezone_offset", 1),
                size=len(file_data),
                encryption_key=base64.b64encode(encryption_key).decode(),
                is_folder=is_folder,
                original_deleted=delete_original
            )
            
            # Add HMAC signature
            metadata.signature = CryptoManager.create_hmac(metadata.to_dict())
            
            # Save metadata
            meta_path = locked_path + ".clocklock"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=4)
            
            if progress_callback:
                progress_callback(0.9, "Cleaning up...")
            
            # Delete original if requested
            if delete_original:
                FileManager.safe_file_operation(os.remove, filepath)
            
            # Save to recent files
            HistoryManager.save_recent_file(locked_path, "locked")
            
            if progress_callback:
                progress_callback(1.0, "Complete!")
            
            logger.info(f"Successfully encrypted file: {filepath}")
            return locked_path
            
        except Exception as e:
            logger.error(f"Failed to encrypt file {filepath}: {e}")
            raise e
    
    @staticmethod
    def _read_file_chunked(filepath: str) -> bytes:
        """Read file in chunks for memory efficiency."""
        data = b""
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                data += chunk
        return data
    
    @staticmethod
    def _write_file_chunked(filepath: str, data: bytes) -> None:
        """Write file in chunks for memory efficiency."""
        with open(filepath, 'wb') as f:
            for i in range(0, len(data), CHUNK_SIZE):
                f.write(data[i:i + CHUNK_SIZE])
    
    @staticmethod
    @handle_exceptions
    def decrypt_file(locked_filepath: str, progress_callback=None) -> Optional[str]:
        """Decrypt a locked file."""
        try:
            if progress_callback:
                progress_callback(0.1, "Reading metadata...")
            
            # Load and verify metadata
            meta_path = locked_filepath + ".clocklock"
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            metadata = FileMetadata.from_dict(metadata_dict)
            
            # Verify HMAC
            if not CryptoManager.verify_hmac(metadata_dict, metadata.signature):
                raise ValueError("File integrity check failed!")
            
            if progress_callback:
                progress_callback(0.2, "Verifying unlock time...")
            
            # Check if file can be unlocked
            unlock_utc = datetime.fromisoformat(metadata.unlock_utc.replace('Z', '+00:00'))
            utc_now, _ = TimeManager.get_internet_date(
                ConfigManager().get("timezone_offset", 1),
                ConfigManager().get("timeout_seconds", 3)
            )
            
            if utc_now and (unlock_utc - utc_now).total_seconds() > 0:
                raise ValueError("File is still locked! Please wait until the unlock time.")
            
            if progress_callback:
                progress_callback(0.3, "Reading encrypted data...")
            
            # Read encrypted data
            encrypted_data = FileEncryption._read_file_chunked(locked_filepath)
            
            if progress_callback:
                progress_callback(0.5, "Decrypting data...")
            
            # Decrypt data
            encryption_key = base64.b64decode(metadata.encryption_key)
            decrypted_data = CryptoManager.decrypt_data(
                encrypted_data, 
                encryption_key,
                lambda p, t: progress_callback(0.5 + p * 0.3, t) if progress_callback else None
            )
            
            if progress_callback:
                progress_callback(0.8, "Restoring file...")
            
            # Determine output path
            directory = Path(locked_filepath).parent
            original_filename = metadata.original_filename
            
            if metadata.is_folder:
                restored_path = FileEncryption._restore_folder(
                    directory, original_filename, decrypted_data, progress_callback
                )
            else:
                restored_path = FileEncryption._restore_file(
                    directory, original_filename, decrypted_data
                )
            
            if progress_callback:
                progress_callback(0.9, "Cleaning up...")
            
            # Clean up locked files
            FileManager.safe_file_operation(os.remove, locked_filepath)
            FileManager.safe_file_operation(os.remove, meta_path)
            
            # Save to history and recent files
            history_entry = {
                "filename": original_filename,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "unlock_time": metadata.unlock_utc,
                "file_type": "folder" if metadata.is_folder else "file"
            }
            HistoryManager.save_history(history_entry)
            HistoryManager.save_recent_file(restored_path, "unlocked")
            
            if progress_callback:
                progress_callback(1.0, "Complete!")
            
            logger.info(f"Successfully decrypted file: {locked_filepath}")
            return restored_path
            
        except Exception as e:
            logger.error(f"Failed to decrypt file {locked_filepath}: {e}")
            raise e
    
    @staticmethod
    def _restore_file(directory: Path, filename: str, data: bytes) -> str:
        """Restore a regular file with conflict resolution."""
        restored_path = directory / filename
        
        # Handle filename conflicts
        counter = 1
        while restored_path.exists():
            name_stem = Path(filename).stem
            suffix = Path(filename).suffix
            restored_path = directory / f"{name_stem}_{counter}{suffix}"
            counter += 1
        
        FileEncryption._write_file_chunked(str(restored_path), data)
        return str(restored_path)
    
    @staticmethod
    def _restore_folder(directory: Path, folder_name: str, zip_data: bytes, progress_callback=None) -> str:
        """Restore a folder from zip data."""
        # Create temporary zip file
        temp_zip = directory / f"{folder_name}_temp.zip"
        FileEncryption._write_file_chunked(str(temp_zip), zip_data)
        
        # Determine extract path with conflict resolution
        extract_path = directory / folder_name
        counter = 1
        while extract_path.exists():
            extract_path = directory / f"{folder_name}_{counter}"
            counter += 1
        
        if progress_callback:
            progress_callback(0.85, "Extracting folder...")
        
        # Extract zip file
        with zipfile.ZipFile(temp_zip, 'r') as zipf:
            zipf.extractall(extract_path)
        
        # Clean up temporary zip
        FileManager.safe_file_operation(os.remove, str(temp_zip))
        
        return str(extract_path)


class BatchProcessor:
    """Handles batch file operations with parallel processing."""
    
    @staticmethod
    def encrypt_batch_files(file_paths: List[str], unlock_utc_time: datetime, 
                          delete_original: bool = True, progress_callback=None) -> Dict[str, int]:
        """Encrypt multiple files in batch with parallel processing."""
        results = {"successful": 0, "failed": 0, "errors": []}
        
        def encrypt_single_file(args):
            filepath, index = args
            try:
                if progress_callback:
                    progress_callback(
                        (index + 0.5) / len(file_paths),
                        f"Encrypting {Path(filepath).name}..."
                    )
                
                FileEncryption.encrypt_file(
                    filepath, unlock_utc_time, 
                    delete_original=delete_original
                )
                return True, None
                
            except Exception as e:
                error_msg = f"Failed to encrypt {Path(filepath).name}: {str(e)}"
                logger.error(error_msg)
                return False, error_msg
        
        # Process files with thread pool for I/O bound operations
        max_workers = min(4, len(file_paths))  # Limit concurrent operations
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            file_args = [(filepath, i) for i, filepath in enumerate(file_paths)]
            
            for i, (success, error) in enumerate(executor.map(encrypt_single_file, file_args)):
                if success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    if error:
                        results["errors"].append(error)
                
                if progress_callback:
                    progress_callback((i + 1) / len(file_paths))
        
        return results


class FolderProcessor:
    """Handles folder compression and processing."""
    
    @staticmethod
    def compress_folder(folder_path: str, progress_callback=None) -> str:
        """Compress folder to temporary zip file."""
        temp_zip = f"{folder_path}_temp.zip"
        
        try:
            # Count total files for progress tracking
            total_files = sum(1 for _, _, files in os.walk(folder_path) for _ in files)
            
            if progress_callback:
                progress_callback(0.1, f"Compressing {total_files} files...")
            
            with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                processed = 0
                
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = Path(root) / file
                        arc_path = file_path.relative_to(folder_path)
                        
                        try:
                            zipf.write(file_path, arc_path)
                            processed += 1
                            
                            if progress_callback and total_files > 0:
                                progress = 0.1 + (processed / total_files) * 0.8
                                progress_callback(
                                    progress,
                                    f"Compressing {processed}/{total_files} files..."
                                )
                                
                        except Exception as e:
                            logger.warning(f"Error compressing {file_path}: {e}")
            
            if progress_callback:
                progress_callback(1.0, "Compression complete!")
            
            return temp_zip
            
        except Exception as e:
            # Clean up on failure
            if Path(temp_zip).exists():
                FileManager.safe_file_operation(os.remove, temp_zip)
            raise e
    
    @staticmethod
    def get_folder_stats(folder_path: str) -> Tuple[int, float, int]:
        """Get folder statistics: file count, size in MB, total size in bytes."""
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = Path(root) / file
                try:
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    file_count += 1
                except (OSError, FileNotFoundError):
                    continue
        
        size_mb = total_size / (1024 * 1024)
        return file_count, size_mb, total_size


class ClockLockGUI:
    """Main GUI application class with improved organization."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.countdown_timer = None
        self.current_file_path = None
        self.selected_files = []
        
        # Initialize GUI
        self._setup_gui()
        self._create_ui_components()
        self._setup_event_handlers()
    
    def _setup_gui(self):
        """Initialize the main GUI window."""
        # Set theme
        theme = "Dark" if self.config_manager.get("dark_mode", 1) else "Light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.app = ctk.CTk()
        self.app.title("ClockLock – Time Lock Tool")
        self.app.geometry("750x800")
        self.app.resizable(False, False)
        
        # Set icon if available
        icon_path = Path("clocklock.ico")
        if icon_path.exists():
            try:
                self.app.iconbitmap(str(icon_path))
            except Exception as e:
                logger.warning(f"Could not set window icon: {e}")
    
    def _create_ui_components(self):
        """Create all UI components."""
        # Define fonts
        self.fonts = {
            'title': ctk.CTkFont("Segoe UI", 28, weight="bold"),
            'subtitle': ctk.CTkFont("Segoe UI", 16),
            'body': ctk.CTkFont("Segoe UI", 14),
            'small': ctk.CTkFont("Segoe UI", 12)
        }
        
        # Create main components
        self._create_title_section()
        self._create_menu_section()
        self._create_main_buttons()
        self._create_info_display()
        self._create_interface_frames()
    
    def _create_title_section(self):
        """Create title and subtitle section."""
        title_card = ctk.CTkFrame(self.app, corner_radius=12)
        title_card.pack(padx=20, pady=(20, 10), fill="x")
        
        ctk.CTkLabel(
            title_card, 
            text="🔒 ClockLock", 
            font=self.fonts['title']
        ).pack(pady=(15, 5))
        
        ctk.CTkLabel(
            title_card, 
            text="Secure time-based file & folder locking with advanced features", 
            font=self.fonts['subtitle']
        ).pack(pady=(0, 15))
    
    def _create_menu_section(self):
        """Create menu bar with settings and history buttons."""
        menu_frame = ctk.CTkFrame(self.app, corner_radius=12)
        menu_frame.pack(padx=20, pady=(0, 10), fill="x")
        
        menu_container = ctk.CTkFrame(menu_frame, fg_color="transparent")
        menu_container.pack(pady=12)
        
        # Menu buttons
        menu_buttons = [
            ("⚙️ Settings", self._open_settings),
            ("📚 History", self._show_history),
            ("📁 Recent", self._show_recent_files),
            ("ℹ️ About", self._show_about)
        ]
        
        for text, command in menu_buttons:
            btn = ctk.CTkButton(
                menu_container, 
                text=text, 
                command=command,
                width=110, 
                height=32, 
                font=self.fonts['small']
            )
            btn.pack(side="left", padx=3)
    
    def _create_main_buttons(self):
        """Create main action buttons."""
        self.main_buttons_frame = ctk.CTkFrame(self.app, corner_radius=12)
        self.main_buttons_frame.pack(padx=20, pady=10, fill="x")
        
        ctk.CTkLabel(
            self.main_buttons_frame, 
            text="Choose an action:", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 12))
        
        # Row 1 - Individual operations
        row1 = ctk.CTkFrame(self.main_buttons_frame, fg_color="transparent")
        row1.pack(pady=(0, 8))
        
        main_actions = [
            ("🔐 Lock File", self._choose_file_to_lock),
            ("🔓 Unlock File", self._choose_file_to_unlock)
        ]
        
        for text, command in main_actions:
            btn = ctk.CTkButton(
                row1, 
                text=text, 
                command=command,
                width=160, 
                height=45, 
                font=self.fonts['body']
            )
            btn.pack(side="left", padx=8)
        
        # Row 2 - Batch operations
        row2 = ctk.CTkFrame(self.main_buttons_frame, fg_color="transparent")
        row2.pack(pady=(0, 15))
        
        batch_actions = [
            ("📦 Batch Lock", self._choose_files_for_batch),
            ("📁 Lock Folder", self._choose_folder_to_lock)
        ]
        
        for text, command in batch_actions:
            btn = ctk.CTkButton(
                row2, 
                text=text, 
                command=command,
                width=160, 
                height=45, 
                font=self.fonts['body']
            )
            btn.pack(side="left", padx=8)
    
    def _create_info_display(self):
        """Create file info display area."""
        info_card = ctk.CTkFrame(self.app, corner_radius=12)
        info_card.pack(padx=20, pady=10, fill="x")
        
        self.file_label = ctk.CTkLabel(
            info_card, 
            text="", 
            font=self.fonts['body'], 
            wraplength=650
        )
        self.file_label.pack(pady=8)
        
        self.current_time_label = ctk.CTkLabel(
            info_card, 
            text="🌍 Internet time will appear here", 
            font=self.fonts['body'], 
            wraplength=650
        )
        self.current_time_label.pack(pady=8)
    
    def _create_interface_frames(self):
        """Create all interface frames (initially hidden)."""
        self._create_lock_frame()
        self._create_unlock_frame()
        self._create_batch_frame()
        self._create_folder_frame()
        self._create_time_selection_frame()
        self._create_countdown_frame()
    
    def _create_lock_frame(self):
        """Create lock file interface."""
        self.lock_frame = ctk.CTkFrame(self.app, corner_radius=12)
        
        ctk.CTkLabel(
            self.lock_frame, 
            text="🔐 Lock File", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(12, 8))
        
        ctk.CTkButton(
            self.lock_frame, 
            text="← Back to Main", 
            command=self._reset_to_main,
            width=130, 
            height=30
        ).pack(pady=(0, 12))
    
    def _create_unlock_frame(self):
        """Create unlock file interface."""
        self.unlock_frame = ctk.CTkFrame(self.app, corner_radius=12)
        
        ctk.CTkLabel(
            self.unlock_frame, 
            text="🔓 Unlock File", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(12, 8))
        
        controls = ctk.CTkFrame(self.unlock_frame, fg_color="transparent")
        controls.pack(pady=(0, 12))
        
        ctk.CTkButton(
            controls, 
            text="← Back to Main", 
            command=self._reset_to_main,
            width=130, 
            height=30
        ).pack(side="left", padx=12)
        
        self.unlock_now_btn = ctk.CTkButton(
            controls, 
            text="🔓 Unlock Now", 
            command=self._unlock_file_now,
            width=150, 
            height=30, 
            state="disabled"
        )
        self.unlock_now_btn.pack(side="right", padx=12)
    
    def _create_batch_frame(self):
        """Create batch processing interface."""
        self.batch_frame = ctk.CTkFrame(self.app, corner_radius=12)
        
        ctk.CTkLabel(
            self.batch_frame, 
            text="📦 Batch Lock Files", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(12, 8))
        
        ctk.CTkButton(
            self.batch_frame, 
            text="← Back to Main", 
            command=self._reset_to_main,
            width=130, 
            height=30
        ).pack(pady=(0, 12))
    
    def _create_folder_frame(self):
        """Create folder locking interface."""
        self.folder_frame = ctk.CTkFrame(self.app, corner_radius=12)
        
        ctk.CTkLabel(
            self.folder_frame, 
            text="📁 Lock Folder", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(12, 8))
        
        ctk.CTkButton(
            self.folder_frame, 
            text="← Back to Main", 
            command=self._reset_to_main,
            width=130, 
            height=30
        ).pack(pady=(0, 12))
    
    def _create_time_selection_frame(self):
        """Create time selection interface."""
        self.time_selection_frame = ctk.CTkFrame(self.app, corner_radius=12)
        
        ctk.CTkLabel(
            self.time_selection_frame, 
            text="⏰ Set Unlock Date & Time", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(12, 8))
        
        # File deletion option
        delete_frame = ctk.CTkFrame(self.time_selection_frame, fg_color="transparent")
        delete_frame.pack(pady=8)
        
        self.delete_original_var = ctk.BooleanVar(
            value=self.config_manager.get("delete_original_files", True)
        )
        ctk.CTkCheckBox(
            delete_frame, 
            text="Delete original files after locking", 
            variable=self.delete_original_var, 
            font=self.fonts['small']
        ).pack()
        
        # Quick presets
        self._create_preset_buttons()
        
        # Date and time selection
        self._create_datetime_selection()
        
        # Encrypt button
        ctk.CTkButton(
            self.time_selection_frame, 
            text="🔐 Encrypt & Lock", 
            command=self._validate_and_encrypt,
            width=220, 
            height=45, 
            font=self.fonts['body']
        ).pack(pady=15)
    
    def _create_preset_buttons(self):
        """Create quick preset buttons."""
        presets_frame = ctk.CTkFrame(self.time_selection_frame, fg_color="transparent")
        presets_frame.pack(pady=8)
        
        ctk.CTkLabel(
            presets_frame, 
            text="Quick presets:", 
            font=self.fonts['small']
        ).pack()
        
        preset_container = ctk.CTkFrame(presets_frame, fg_color="transparent")
        preset_container.pack(pady=5)
        
        presets = self.config_manager.get("quick_presets", {
            "1 Hour": 1, "1 Day": 24, "1 Week": 168, "1 Month": 720
        })
        
        for preset_name, hours in presets.items():
            btn = ctk.CTkButton(
                preset_container, 
                text=preset_name,
                command=lambda h=hours: self._apply_preset(h),
                width=85, 
                height=28, 
                font=ctk.CTkFont(size=10)
            )
            btn.pack(side="left", padx=3)
    
    def _create_datetime_selection(self):
        """Create date and time selection widgets."""
        # Date picker
        date_frame = ctk.CTkFrame(self.time_selection_frame, fg_color="transparent")
        date_frame.pack(pady=8)
        
        ctk.CTkLabel(
            date_frame, 
            text="📅 Date:", 
            font=self.fonts['body']
        ).pack(side="left", padx=(0, 12))
        
        self.date_picker = DateEntry(
            date_frame, 
            width=12, 
            background='darkblue',
            foreground='white', 
            borderwidth=2, 
            date_pattern='yyyy-mm-dd'
        )
        self.date_picker.pack(side="left")
        
        # Time entry
        time_frame = ctk.CTkFrame(self.time_selection_frame, fg_color="transparent")
        time_frame.pack(pady=8)
        
        ctk.CTkLabel(
            time_frame, 
            text="🕐 Time (24h):", 
            font=self.fonts['body']
        ).pack(side="left", padx=(0, 12))
        
        self.time_entry = ctk.CTkEntry(
            time_frame, 
            width=110, 
            placeholder_text="14:30"
        )
        self.time_entry.pack(side="left")
    
    def _create_countdown_frame(self):
        """Create countdown display frame."""
        self.countdown_frame = ctk.CTkFrame(self.app, corner_radius=12)
        
        self.countdown_label = ctk.CTkLabel(
            self.countdown_frame, 
            text="", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.countdown_label.pack(pady=25)
    
    def _setup_event_handlers(self):
        """Setup keyboard shortcuts and other event handlers."""
        # Keyboard shortcuts
        self.app.bind('<Control-o>', lambda e: self._choose_file_to_lock())
        self.app.bind('<Control-u>', lambda e: self._choose_file_to_unlock())
        self.app.bind('<Control-b>', lambda e: self._choose_files_for_batch())
        self.app.bind('<Control-f>', lambda e: self._choose_folder_to_lock())
        self.app.bind('<Escape>', lambda e: self._reset_to_main())
        
        # Window close handler
        self.app.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _on_closing(self):
        """Handle application closing."""
        if self.countdown_timer:
            self.countdown_timer.stop()
        self.app.destroy()
    
    # UI State Management
    def _reset_to_main(self):
        """Reset UI to main menu."""
        if self.countdown_timer:
            self.countdown_timer.stop()
            self.countdown_timer = None
        
        self.current_file_path = None
        self.selected_files = []
        
        # Hide all secondary frames
        for frame in [self.lock_frame, self.unlock_frame, self.time_selection_frame, 
                     self.countdown_frame, self.batch_frame, self.folder_frame]:
            frame.pack_forget()
        
        # Show main buttons
        self.main_buttons_frame.pack(padx=20, pady=10, fill="x")
        
        # Reset labels
        self.file_label.configure(text="")
        self.current_time_label.configure(text="🌍 Internet time will appear here")
        self.countdown_label.configure(text="")
    
    def _show_lock_interface(self):
        """Show the lock file interface."""
        self._update_file_display()
        self._update_current_time()
        
        # Hide main buttons and show lock interface
        self.main_buttons_frame.pack_forget()
        self.lock_frame.pack(padx=20, pady=10, fill="x")
        self.time_selection_frame.pack(padx=20, pady=10, fill="x")
    
    def _show_unlock_interface(self, metadata: FileMetadata):
        """Show the unlock file interface."""
        self._update_file_display()
        
        # Hide main buttons and show unlock interface
        self.main_buttons_frame.pack_forget()
        self.unlock_frame.pack(padx=20, pady=10, fill="x")
        self.countdown_frame.pack(padx=20, pady=10, fill="x")
        
        # Start countdown or enable unlock
        self._setup_unlock_countdown(metadata)
    
    def _setup_unlock_countdown(self, metadata: FileMetadata):
        """Setup countdown timer for unlock."""
        unlock_utc = datetime.fromisoformat(metadata.unlock_utc.replace('Z', '+00:00'))
        utc_now, _ = TimeManager.get_internet_date(
            self.config_manager.get("timezone_offset", 1),
            self.config_manager.get("timeout_seconds", 3)
        )
        
        if not utc_now:
            messagebox.showerror("Error", "Cannot verify time - no internet connection!")
            return
        
        time_remaining = (unlock_utc - utc_now).total_seconds()
        
        if time_remaining <= 0:
            self.countdown_label.configure(
                text="✅ File can now be unlocked!",
                text_color="#00ff00"
            )
            self.unlock_now_btn.configure(state="normal")
        else:
            self.countdown_label.configure(text_color="#ffffff")
            self.unlock_now_btn.configure(state="disabled")
            self.countdown_timer = CountdownTimer(
                self.countdown_label, 
                unlock_utc, 
                lambda: self.unlock_now_btn.configure(state="normal")
            )
            self.countdown_timer.start()
    
    def _update_file_display(self):
        """Update file information display."""
        if not self.current_file_path and not self.selected_files:
            return
        
        if self.selected_files:
            # Batch display
            total_size = sum(Path(f).stat().st_size for f in self.selected_files 
                           if Path(f).exists())
            size_str = FileManager.format_file_size(total_size)
            self.file_label.configure(
                text=f"📦 {len(self.selected_files)} files selected ({size_str})"
            )
        elif self.current_file_path:
            # Single file display
            file_path = Path(self.current_file_path)
            if file_path.exists():
                file_size = FileManager.format_file_size(file_path.stat().st_size)
                file_icon = FileManager.get_file_icon(str(file_path))
                short_name = FileManager.shorten_filename(
                    file_path.name, 
                    self.config_manager.get("max_filename_length", 40)
                )
                self.file_label.configure(text=f"{file_icon} {short_name} ({file_size})")
    
    def _update_current_time(self):
        """Update current time display."""
        utc, local = TimeManager.get_internet_date(
            self.config_manager.get("timezone_offset", 1),
            self.config_manager.get("timeout_seconds", 3)
        )
        
        if local:
            timezone_str = self.config_manager.get("selected_timezone", "UTC+1")
            time_str = local.strftime('%Y-%m-%d %H:%M:%S')
            self.current_time_label.configure(
                text=f"🌍 Current Time ({timezone_str}):\n{time_str}"
            )
        else:
            self.current_time_label.configure(
                text="❌ Failed to fetch current time. Check your internet connection."
            )
    
    # File Selection Handlers
    @handle_exceptions
    def _choose_file_to_lock(self):
        """Handle file selection for locking."""
        selected = filedialog.askopenfilename(title="Select file to lock")
        if not selected:
            return
        
        self._process_file_for_locking(selected)
    
    def _process_file_for_locking(self, filepath: str):
        """Process selected file for locking."""
        # Check if file is already locked
        if filepath.endswith('.locked'):
            messagebox.showerror(
                "Error", 
                "This file is already locked! Use 'Unlock a File' instead."
            )
            return
        
        # Check file size
        file_size_mb = FileManager.get_file_size_mb(filepath)
        max_size = self.config_manager.get("max_file_size_mb", 500)
        
        if file_size_mb > max_size:
            if not messagebox.askyesno(
                "Large File Warning", 
                f"⚠️ File size ({file_size_mb:.1f} MB) exceeds limit ({max_size} MB).\n\n"
                "Do you want to continue?"
            ):
                return
        
        self.current_file_path = filepath
        self._show_lock_interface()
    
    @handle_exceptions
    def _choose_file_to_unlock(self):
        """Handle file selection for unlocking."""
        selected = filedialog.askopenfilename(
            title="Select .locked file to unlock",
            filetypes=[("Locked files", "*.locked"), ("All files", "*.*")]
        )
        if not selected:
            return
        
        self._choose_specific_file_to_unlock(selected)
    
    def _choose_specific_file_to_unlock(self, filepath: str):
        """Process specific locked file for unlocking."""
        if not filepath.endswith('.locked'):
            messagebox.showerror("Error", "Please select a .locked file!")
            return
        
        meta_path = filepath + ".clocklock"
        if not Path(meta_path).exists():
            messagebox.showerror(
                "Error", 
                "No .clocklock metadata file found!\nThis file cannot be unlocked."
            )
            return
        
        try:
            # Load and verify metadata
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            if not CryptoManager.verify_hmac(metadata_dict, metadata_dict.get('signature', '')):
                messagebox.showerror(
                    "Error", 
                    "File integrity check failed!\n"
                    "The metadata file may be corrupted or tampered with."
                )
                return
            
            metadata = FileMetadata.from_dict(metadata_dict)
            self.current_file_path = filepath
            self._show_unlock_interface(metadata)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read metadata: {str(e)}")
    
    @handle_exceptions
    def _choose_files_for_batch(self):
        """Handle multiple file selection for batch processing."""
        files = filedialog.askopenfilenames(title="Select multiple files to lock")
        if not files:
            return
        
        self.selected_files = list(files)
        
        # Check total size
        total_size = sum(FileManager.get_file_size_mb(f) for f in self.selected_files)
        max_size = self.config_manager.get("max_file_size_mb", 500)
        
        if total_size > max_size:
            if not messagebox.askyesno(
                "Large Batch Warning", 
                f"⚠️ Total size ({total_size:.1f} MB, {len(self.selected_files)} files) "
                f"exceeds limit ({max_size} MB).\n\nDo you want to continue?"
            ):
                return
        
        self._show_batch_interface()
    
    def _show_batch_interface(self):
        """Show batch processing interface."""
        self.main_buttons_frame.pack_forget()
        self.batch_frame.pack(padx=20, pady=10, fill="x")
        self.time_selection_frame.pack(padx=20, pady=10, fill="x")
        self._update_file_display()
    
    @handle_exceptions
    def _choose_folder_to_lock(self):
        """Handle folder selection for locking."""
        selected = filedialog.askdirectory(title="Select folder to lock")
        if not selected:
            return
        
        # Get folder statistics
        file_count, size_mb, total_size = FolderProcessor.get_folder_stats(selected)
        max_size = self.config_manager.get("max_file_size_mb", 500)
        
        if size_mb > max_size:
            if not messagebox.askyesno(
                "Large Folder Warning", 
                f"⚠️ Folder size ({size_mb:.1f} MB, {file_count} files) "
                f"exceeds limit ({max_size} MB).\n\nDo you want to continue?"
            ):
                return
        
        # Show progress and compress folder
        progress_dialog = ProgressDialog(self.app, "Compressing Folder", "Preparing folder...")
        
        try:
            temp_zip = FolderProcessor.compress_folder(
                selected, 
                progress_dialog.update_progress
            )
            progress_dialog.close()
            
            self.current_file_path = temp_zip
            self._show_folder_interface(selected, file_count, size_mb)
            
        except Exception as e:
            progress_dialog.close()
            messagebox.showerror("Error", f"Failed to compress folder: {str(e)}")
    
    def _show_folder_interface(self, original_folder: str, file_count: int, size_mb: float):
        """Show folder locking interface."""
        folder_name = Path(original_folder).name
        
        self.main_buttons_frame.pack_forget()
        self.folder_frame.pack(padx=20, pady=10, fill="x")
        self.time_selection_frame.pack(padx=20, pady=10, fill="x")
        
        self.file_label.configure(
            text=f"📁 {folder_name} ({file_count} files, {size_mb:.1f} MB)"
        )
        self._update_current_time()
    
    # Time and Encryption Handlers
    def _apply_preset(self, hours: int):
        """Apply quick time preset."""
        future_time = datetime.now() + timedelta(hours=hours)
        
        # Update date picker
        self.date_picker.set_date(future_time.date())
        
        # Update time entry
        self.time_entry.delete(0, 'end')
        self.time_entry.insert(0, future_time.strftime("%H:%M"))
    
    @handle_exceptions
    def _validate_and_encrypt(self):
        """Validate inputs and start encryption process."""
        if not self.current_file_path and not self.selected_files:
            messagebox.showerror("Error", "No file selected!")
            return
        
        try:
            # Get selected date and time
            selected_date = self.date_picker.get_date()
            time_str = self.time_entry.get().strip()
            
            if not time_str:
                messagebox.showerror("Error", "Please enter a time (HH:MM format)!")
                return
            
            # Parse and validate time
            try:
                hour, minute = map(int, time_str.split(':'))
                if not (0 <= hour <= 23 and 0 <= minute <= 59):
                    raise ValueError()
            except ValueError:
                messagebox.showerror("Error", "Invalid time format! Use HH:MM (24-hour format)")
                return
            
            # Create datetime object
            local_unlock_time = datetime.combine(
                selected_date, 
                datetime.min.time().replace(hour=hour, minute=minute)
            )
            local_timezone = timezone(timedelta(hours=self.config_manager.get("timezone_offset", 1)))
            local_unlock_time = local_unlock_time.replace(tzinfo=local_timezone)
            
            # Convert to UTC
            utc_unlock_time = local_unlock_time.astimezone(timezone.utc)
            
            # Validate time is in future
            utc_now, _ = TimeManager.get_internet_date(
                self.config_manager.get("timezone_offset", 1),
                self.config_manager.get("timeout_seconds", 3)
            )
            
            if not utc_now:
                messagebox.showerror("Error", "Cannot validate time - no internet connection!")
                return
            
            if utc_unlock_time <= utc_now:
                messagebox.showerror(
                    "Error", 
                    "Selected unlock time is in the past!\nPlease choose a future date and time."
                )
                return
            
            # Get delete preference
            delete_files = self.delete_original_var.get()
            
            # Start encryption process
            self._start_encryption(utc_unlock_time, delete_files)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def _start_encryption(self, unlock_utc_time: datetime, delete_original: bool):
        """Start the encryption process."""
        try:
            if self.selected_files:
                # Batch encryption
                self._encrypt_batch_files(unlock_utc_time, delete_original)
            elif self.current_file_path:
                # Single file/folder encryption
                is_folder = self.current_file_path.endswith("_temp.zip")
                self._encrypt_single_file(unlock_utc_time, is_folder, delete_original)
            
        except Exception as e:
            logger.error(f"Encryption process failed: {e}")
            messagebox.showerror("Error", f"Encryption failed: {str(e)}")
    
    def _encrypt_single_file(self, unlock_utc_time: datetime, is_folder: bool, delete_original: bool):
        """Encrypt a single file with progress tracking."""
        progress_dialog = ProgressDialog(self.app, "Encrypting File", "Starting encryption...")
        
        try:
            locked_path = FileEncryption.encrypt_file(
                self.current_file_path,
                unlock_utc_time,
                is_folder=is_folder,
                delete_original=delete_original,
                progress_callback=progress_dialog.update_progress
            )
            
            progress_dialog.close()
            
            if locked_path:
                self._show_encryption_success(locked_path, delete_original, unlock_utc_time)
            
        except Exception as e:
            progress_dialog.close()
            raise e
    
    def _encrypt_batch_files(self, unlock_utc_time: datetime, delete_original: bool):
        """Encrypt multiple files with progress tracking."""
        progress_dialog = ProgressDialog(self.app, "Batch Encryption", "Starting batch encryption...")
        
        try:
            results = BatchProcessor.encrypt_batch_files(
                self.selected_files,
                unlock_utc_time,
                delete_original=delete_original,
                progress_callback=progress_dialog.update_progress
            )
            
            progress_dialog.close()
            
            self._show_batch_results(results, delete_original, unlock_utc_time)
            
        except Exception as e:
            progress_dialog.close()
            raise e
    
    def _show_encryption_success(self, locked_path: str, delete_original: bool, unlock_utc_time: datetime):
        """Show encryption success and start countdown."""
        # Hide current interfaces
        self.lock_frame.pack_forget()
        self.folder_frame.pack_forget()
        self.time_selection_frame.pack_forget()
        
        # Show countdown
        self.countdown_frame.pack(padx=20, pady=10, fill="x")
        
        # Update display
        status = "encrypted and original deleted" if delete_original else "encrypted (original kept)"
        filename = Path(locked_path).name
        self.file_label.configure(text=f"🔐 File {status}: {filename}")
        
        # Start countdown timer
        self.countdown_timer = CountdownTimer(self.countdown_label, unlock_utc_time)
        self.countdown_timer.start()
        
        # Show success message
        success_msg = f"File has been {status} successfully!\nLocked file: {filename}"
        messagebox.showinfo("Success", success_msg)
    
    def _show_batch_results(self, results: Dict[str, int], delete_original: bool, unlock_utc_time: datetime):
        """Show batch encryption results and start countdown."""
        # Hide current interfaces
        self.batch_frame.pack_forget()
        self.time_selection_frame.pack_forget()
        
        # Show countdown
        self.countdown_frame.pack(padx=20, pady=10, fill="x")
        
        # Update display
        status = "encrypted" + (" (originals deleted)" if delete_original else " (originals kept)")
        self.file_label.configure(text=f"🔐 Batch {status}: {results['successful']} files")
        
        # Start countdown timer
        self.countdown_timer = CountdownTimer(self.countdown_label, unlock_utc_time)
        self.countdown_timer.start()
        
        # Show results message
        result_message = f"Successfully encrypted: {results['successful']} files\nFailed: {results['failed']} files"
        if not delete_original:
            result_message += "\n\nOriginal files were kept alongside locked files."
        
        if results['errors']:
            result_message += f"\n\nErrors:\n" + "\n".join(results['errors'][:5])
            if len(results['errors']) > 5:
                result_message += f"\n... and {len(results['errors']) - 5} more errors."
        
        messagebox.showinfo("Batch Encryption Complete", result_message)
    
    # Unlock Handler
    @handle_exceptions
    def _unlock_file_now(self):
        """Unlock the currently selected file."""
        if not self.current_file_path:
            messagebox.showerror("Error", "No file selected!")
            return
        
        progress_dialog = ProgressDialog(self.app, "Unlocking File", "Starting decryption...")
        
        try:
            restored_path = FileEncryption.decrypt_file(
                self.current_file_path,
                progress_callback=progress_dialog.update_progress
            )
            
            progress_dialog.close()
            
            if restored_path:
                # Check for auto-relock
                if self.config_manager.get("auto_relock", False):
                    if messagebox.askyesno("Auto-Relock", "Auto-relock is enabled. Lock this file again?"):
                        self.current_file_path = restored_path
                        self._show_lock_interface()
                        return
                
                filename = Path(restored_path).name
                messagebox.showinfo("Success", f"File unlocked successfully!\nRestored to: {filename}")
                self._reset_to_main()
            
        except Exception as e:
            progress_dialog.close()
            messagebox.showerror("Error", f"Unlocking failed: {str(e)}")
    
    # Settings and Info Windows
    def _open_settings(self):
        """Open settings window."""
        SettingsWindow(self.app, self.config_manager)
    
    def _show_history(self):
        """Show unlock history window."""
        HistoryWindow(self.app)
    
    def _show_recent_files(self):
        """Show recent files window."""
        RecentFilesWindow(self.app, self._choose_specific_file_to_unlock)
    
    def _show_about(self):
        """Show about dialog."""
        about_text = """ClockLock v1.0

A secure time-based file locking application with advanced features:
• Individual file and folder locking
• Batch processing capabilities
• Time-based unlock mechanism
• Internet time synchronization
• HMAC integrity verification
• Comprehensive history tracking

Keyboard Shortcuts:
• Ctrl+O: Lock File
• Ctrl+U: Unlock File  
• Ctrl+B: Batch Lock
• Ctrl+F: Lock Folder
• Escape: Back to Main

© 2025 ClockLock"""
        
        messagebox.showinfo("About ClockLock", about_text)
    
    def run(self):
        """Start the GUI application."""
        logger.info("Starting ClockLock application")
        self.app.mainloop()


class SettingsWindow:
    """Settings configuration window."""
    
    def __init__(self, parent, config_manager: ConfigManager):
        self.parent = parent
        self.config_manager = config_manager
        self.config = config_manager.load_config()
        
        self._create_window()
        self._create_settings_ui()
    
    def _create_window(self):
        """Create settings window."""
        self.window = ctk.CTkToplevel(self.parent)
        self.window.title("⚙️ Settings")
        self.window.geometry("550x700")
        self.window.resizable(False, False)
        self.window.transient(self.parent)
        self.window.grab_set()
        
        # Center window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (550 // 2)
        y = (self.window.winfo_screenheight() // 2) - (700 // 2)
        self.window.geometry(f"550x700+{x}+{y}")
    
    def _create_settings_ui(self):
        """Create settings UI components."""
        # Create scrollable frame
        main_frame = ctk.CTkScrollableFrame(self.window)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Theme settings
        self._create_theme_section(main_frame)
        
        # File handling settings
        self._create_file_section(main_frame)
        
        # Sound settings
        self._create_sound_section(main_frame)
        
        # Performance settings
        self._create_performance_section(main_frame)
        
        # Security settings
        self._create_security_section(main_frame)
        
        # Timezone settings
        self._create_timezone_section(main_frame)
        
        # Save button
        ctk.CTkButton(
            self.window, 
            text="💾 Save Settings", 
            command=self._save_settings,
            width=200,
            height=40
        ).pack(pady=20)
    
    def _create_theme_section(self, parent):
        """Create theme settings section."""
        theme_frame = ctk.CTkFrame(parent)
        theme_frame.pack(padx=10, pady=10, fill="x")
        
        ctk.CTkLabel(
            theme_frame, 
            text="🎨 Theme Settings", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.theme_var = ctk.StringVar(value="Dark" if self.config.get("dark_mode", 1) else "Light")
        theme_menu = ctk.CTkOptionMenu(
            theme_frame, 
            values=["Dark", "Light"], 
            variable=self.theme_var
        )
        theme_menu.pack(pady=5)
    
    def _create_file_section(self, parent):
        """Create file handling settings section."""
        file_frame = ctk.CTkFrame(parent)
        file_frame.pack(padx=10, pady=10, fill="x")
        
        ctk.CTkLabel(
            file_frame, 
            text="📁 File Handling", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.delete_original_var = ctk.BooleanVar(value=self.config.get("delete_original_files", True))
        delete_check = ctk.CTkCheckBox(
            file_frame, 
            text="Delete original files after locking (default)", 
            variable=self.delete_original_var
        )
        delete_check.pack(pady=5)
        
        ctk.CTkLabel(
            file_frame, 
            text="When disabled, original files will be kept alongside locked files", 
            font=ctk.CTkFont(size=10), 
            text_color="gray"
        ).pack(pady=(0, 10))
        
        # File size limit
        size_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        size_frame.pack(pady=5, fill="x")
        
        ctk.CTkLabel(size_frame, text="Max file size (MB):", font=ctk.CTkFont(size=12)).pack(side="left")
        self.size_var = ctk.StringVar(value=str(self.config.get("max_file_size_mb", 500)))
        size_entry = ctk.CTkEntry(size_frame, textvariable=self.size_var, width=100)
        size_entry.pack(side="right", padx=10)
    
    def _create_sound_section(self, parent):
        """Create sound settings section."""
        sound_frame = ctk.CTkFrame(parent)
        sound_frame.pack(padx=10, pady=10, fill="x")
        
        ctk.CTkLabel(
            sound_frame, 
            text="🔊 Sound Settings", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.sound_var = ctk.BooleanVar(value=self.config.get("sound_notifications", True))
        sound_check = ctk.CTkCheckBox(
            sound_frame, 
            text="Enable sound notifications", 
            variable=self.sound_var
        )
        sound_check.pack(pady=5)
        
        if not SOUND_AVAILABLE:
            sound_check.configure(state="disabled")
            ctk.CTkLabel(
                sound_frame, 
                text="Sound unavailable (pygame not installed)", 
                font=ctk.CTkFont(size=10), 
                text_color="orange"
            ).pack()
    
    def _create_performance_section(self, parent):
        """Create performance settings section."""
        perf_frame = ctk.CTkFrame(parent)
        perf_frame.pack(padx=10, pady=10, fill="x")
        
        ctk.CTkLabel(
            perf_frame, 
            text="⚡ Performance", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Timeout setting
        timeout_frame = ctk.CTkFrame(perf_frame, fg_color="transparent")
        timeout_frame.pack(pady=5, fill="x")
        
        ctk.CTkLabel(timeout_frame, text="Network timeout (seconds):", font=ctk.CTkFont(size=12)).pack(side="left")
        self.timeout_var = ctk.StringVar(value=str(self.config.get("timeout_seconds", 3)))
        timeout_entry = ctk.CTkEntry(timeout_frame, textvariable=self.timeout_var, width=100)
        timeout_entry.pack(side="right", padx=10)
        
        # Filename length
        filename_frame = ctk.CTkFrame(perf_frame, fg_color="transparent")
        filename_frame.pack(pady=5, fill="x")
        
        ctk.CTkLabel(filename_frame, text="Max filename display length:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.filename_var = ctk.StringVar(value=str(self.config.get("max_filename_length", 40)))
        filename_entry = ctk.CTkEntry(filename_frame, textvariable=self.filename_var, width=100)
        filename_entry.pack(side="right", padx=10)
    
    def _create_security_section(self, parent):
        """Create security settings section."""
        security_frame = ctk.CTkFrame(parent)
        security_frame.pack(padx=10, pady=10, fill="x")
        
        ctk.CTkLabel(
            security_frame, 
            text="🔒 Security", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.auto_relock_var = ctk.BooleanVar(value=self.config.get("auto_relock", False))
        relock_check = ctk.CTkCheckBox(
            security_frame, 
            text="Automatically re-lock files after unlock", 
            variable=self.auto_relock_var
        )
        relock_check.pack(pady=5)
    
    def _create_timezone_section(self, parent):
        """Create timezone settings section."""
        tz_frame = ctk.CTkFrame(parent)
        tz_frame.pack(padx=10, pady=10, fill="x")
        
        ctk.CTkLabel(
            tz_frame, 
            text="🌍 Timezone", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        timezones = [f"UTC{i:+d}" for i in range(-12, 13)]
        self.tz_var = ctk.StringVar(value=self.config.get("selected_timezone", "UTC+1"))
        tz_menu = ctk.CTkOptionMenu(tz_frame, values=timezones, variable=self.tz_var)
        tz_menu.pack(pady=5)
    
    def _save_settings(self):
        """Save all settings."""
        try:
            # Update configuration
            self.config["dark_mode"] = 1 if self.theme_var.get() == "Dark" else 0
            self.config["sound_notifications"] = self.sound_var.get()
            self.config["auto_relock"] = self.auto_relock_var.get()
            self.config["delete_original_files"] = self.delete_original_var.get()
            self.config["selected_timezone"] = self.tz_var.get()
            
            # Validate and set numeric values
            try:
                self.config["max_file_size_mb"] = max(1, int(self.size_var.get()))
                self.config["timeout_seconds"] = max(1, int(self.timeout_var.get()))
                self.config["max_filename_length"] = max(10, int(self.filename_var.get()))
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numeric values.")
                return
            
            # Extract timezone offset
            tz_str = self.tz_var.get()
            self.config["timezone_offset"] = int(tz_str.replace("UTC", ""))
            
            # Save configuration
            self.config_manager.save_config(self.config)
            
            # Apply theme immediately
            theme = "Dark" if self.config["dark_mode"] else "Light"
            ctk.set_appearance_mode(theme)
            
            messagebox.showinfo("Settings", "Settings saved successfully!")
            self.window.destroy()
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")


class HistoryWindow:
    """History display window."""
    
    def __init__(self, parent):
        self.parent = parent
        self._create_window()
        self._populate_history()
    
    def _create_window(self):
        """Create history window."""
        self.window = ctk.CTkToplevel(self.parent)
        self.window.title("📚 Unlock History")
        self.window.geometry("650x450")
        self.window.transient(self.parent)
        
        # Center window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (650 // 2)
        y = (self.window.winfo_screenheight() // 2) - (450 // 2)
        self.window.geometry(f"650x450+{x}+{y}")
    
    def _populate_history(self):
        """Populate history list."""
        scrollable_frame = ctk.CTkScrollableFrame(self.window)
        scrollable_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        history = HistoryManager.load_history()
        
        if not history:
            ctk.CTkLabel(
                scrollable_frame, 
                text="No unlock history found.", 
                font=ctk.CTkFont(size=14)
            ).pack(pady=50)
            return
        
        for entry in history:
            entry_frame = ctk.CTkFrame(scrollable_frame)
            entry_frame.pack(padx=10, pady=5, fill="x")
            
            filename = entry.get("filename", "Unknown file")
            timestamp = entry.get("timestamp", "Unknown time")
            file_type = entry.get("file_type", "file")
            
            icon = "📁" if file_type == "folder" else "🔓"
            
            ctk.CTkLabel(
                entry_frame, 
                text=f"{icon} {filename}", 
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(anchor="w", padx=10, pady=(5, 2))
            
            ctk.CTkLabel(
                entry_frame, 
                text=f"Unlocked: {timestamp}", 
                font=ctk.CTkFont(size=10)
            ).pack(anchor="w", padx=10, pady=(0, 5))


class RecentFilesWindow:
    """Recent files display window."""
    
    def __init__(self, parent, unlock_callback):
        self.parent = parent
        self.unlock_callback = unlock_callback
        self._create_window()
        self._populate_recent_files()
    
    def _create_window(self):
        """Create recent files window."""
        self.window = ctk.CTkToplevel(self.parent)
        self.window.title("📁 Recent Files")
        self.window.geometry("650x450")
        self.window.transient(self.parent)
        
        # Center window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (650 // 2)
        y = (self.window.winfo_screenheight() // 2) - (450 // 2)
        self.window.geometry(f"650x450+{x}+{y}")
    
    def _populate_recent_files(self):
        """Populate recent files list."""
        scrollable_frame = ctk.CTkScrollableFrame(self.window)
        scrollable_frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        recent = HistoryManager.load_recent_files()
        
        if not recent:
            ctk.CTkLabel(
                scrollable_frame, 
                text="No recent files found.", 
                font=ctk.CTkFont(size=14)
            ).pack(pady=50)
            return
        
        for entry in recent:
            entry_frame = ctk.CTkFrame(scrollable_frame)
            entry_frame.pack(padx=10, pady=5, fill="x")
            
            filename = entry.get("filename", "Unknown file")
            action = entry.get("action", "Unknown")
            timestamp = entry.get("timestamp", "Unknown time")
            filepath = entry.get("path", "")
            
            icon = "🔐" if action == "locked" else "🔓"
            
            # File info
            info_frame = ctk.CTkFrame(entry_frame, fg_color="transparent")
            info_frame.pack(fill="x", padx=10, pady=5)
            
            ctk.CTkLabel(
                info_frame, 
                text=f"{icon} {filename}", 
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(anchor="w")
            
            ctk.CTkLabel(
                info_frame, 
                text=f"{action.title()}: {timestamp}", 
                font=ctk.CTkFont(size=10)
            ).pack(anchor="w")
            
            # Action button
            if Path(filepath).exists():
                def create_handler(path=filepath):
                    if action == "locked" and path.endswith('.locked'):
                        return lambda: (self.unlock_callback(path), self.window.destroy())
                    else:
                        return lambda: messagebox.showinfo("File Info", f"File location: {path}")
                
                btn_text = "Unlock" if action == "locked" else "Show"
                ctk.CTkButton(
                    entry_frame, 
                    text=btn_text, 
                    command=create_handler(),
                    width=80, 
                    height=25
                ).pack(anchor="e", padx=10, pady=(0, 5))


def main():
    """Main application entry point."""
    try:
        # Create and run the GUI application
        app = ClockLockGUI()
        app.run()
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"Error: {e}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()