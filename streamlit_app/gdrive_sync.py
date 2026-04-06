"""
Google Drive Sync Utility for NetGuard
Fetches benchmark results from Google Colab runs stored in Google Drive
"""

import os
import json
import zipfile
from pathlib import Path
from typing import Optional, Dict, List
import streamlit as st

try:
    from google.auth.transport.requests import Request
    from google.oauth2.service_account import Credentials
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import io
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False

class GoogleDriveSync:
    """Manages synchronization with Google Drive for Colab results."""
    
    def __init__(self):
        self.drive_service = None
        self.results_folder_name = "NetGuard_Results"
        self.is_authenticated = False
    
    def authenticate_with_streamlit(self):
        """Authenticate using Streamlit's Google OAuth integration."""
        try:
            # This would use Streamlit's built-in Google authentication
            # when available in future versions
            st.warning("""
            Google Drive sync requires manual authentication.
            \nAlternatively, you can:
            1. Download results manually from Google Drive
            2. Extract to `backend/results/`
            3. Refresh Dashboard
            """)
            return False
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            return False
    
    def find_results_folder(self) -> Optional[str]:
        """Find the NetGuard_Results folder in Google Drive."""
        if not self.drive_service:
            return None
        
        try:
            query = f"name='{self.results_folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)',
                pageSize=10
            ).execute()
            
            files = results.get('files', [])
            if files:
                return files[0]['id']  # Return first matching folder
            return None
        except Exception as e:
            st.error(f"Error finding folder: {e}")
            return None
    
    def list_results(self, folder_id: str) -> List[Dict]:
        """List all result files in the NetGuard_Results folder."""
        if not self.drive_service:
            return []
        
        try:
            query = f"parents='{folder_id}' and trashed=false"
            results = self.drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType, modifiedTime)',
                pageSize=100
            ).execute()
            
            return results.get('files', [])
        except Exception as e:
            st.error(f"Error listing files: {e}")
            return []
    
    def download_file(self, file_id: str, file_name: str, output_path: Path) -> bool:
        """Download a file from Google Drive."""
        if not self.drive_service:
            return False
        
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            file_handle = io.FileIO(str(output_path / file_name), 'wb')
            downloader = MediaIoBaseDownload(file_handle, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            return True
        except Exception as e:
            st.error(f"Download failed: {e}")
            return False
    
    def sync_results(self, output_dir: Path) -> bool:
        """Download all results from Google Drive."""
        if not self.drive_service:
            return False
        
        # Find results folder
        folder_id = self.find_results_folder()
        if not folder_id:
            st.warning("NetGuard_Results folder not found in Google Drive")
            return False
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # List and download files
        files = self.list_results(folder_id)
        if not files:
            st.warning("No files found in NetGuard_Results folder")
            return False
        
        st.info(f"Found {len(files)} files to download...")
        
        with st.spinner("Downloading from Google Drive..."):
            for file_info in files:
                success = self.download_file(
                    file_info['id'],
                    file_info['name'],
                    output_dir
                )
                if success:
                    st.success(f"✅ Downloaded: {file_info['name']}")
                else:
                    st.error(f"❌ Failed to download: {file_info['name']}")
        
        return True

def display_manual_sync_instructions():
    """Show step-by-step manual sync instructions."""
    st.markdown("""
    ## 📥 Manual Results Sync from Google Drive
    
    ### Step 1: Find Results in Google Drive
    1. Go to [Google Drive](https://drive.google.com)
    2. Look for folder named `NetGuard_Results`
    3. Double-click to open it
    
    ### Step 2: Download Results
    1. Select ALL files in the folder (Ctrl+A)
    2. Right-click → Download
    3. This will download a ZIP file
    
    ### Step 3: Extract Files
    1. Extract the ZIP file locally
    2. You should see:
       - `metrics/` folder with `all_results.json`
       - `plots/` folder with PNG visualizations
    
    ### Step 4: Upload to Your Machine
    1. Copy the contents to your project:
       ```
       cp -r path/to/extracted/* backend/results/
       ```
    2. Or manually:
       - Copy `metrics/` to `backend/results/`
       - Copy `plots/` to `backend/results/`
    
    ### Step 5: Refresh Dashboard
    1. Refresh the Streamlit Dashboard
    2. Results will automatically appear!
    """)

def display_sync_section():
    """Display the results sync section in the UI."""
    st.markdown("---")
    st.markdown("## 📥 Sync Results from Colab")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Option 1: Automatic Sync (Coming Soon)")
        if st.button("🔗 Connect Google Drive", use_container_width=True):
            st.info("""
            This feature requires Google OAuth authentication.
            Coming in next update!
            \nFor now, use manual sync below.
            """)
    
    with col2:
        st.markdown("### Option 2: Manual Sync")
        if st.button("📖 Show Instructions", use_container_width=True):
            with st.expander("Manual Sync Steps", expanded=True):
                display_manual_sync_instructions()
