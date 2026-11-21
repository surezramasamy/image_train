"""Channel between the local tool and the Colab worker: a 'YoloStudio/jobs/<job_id>/' folder on Google Drive.
Two backends:
  * FSTransport    - Google Drive for Desktop synced folder (no OAuth setup needed)
  * DriveTransport - Google Drive API (needs client_secret.json, one-time browser sign-in)
"""
import io, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config.json"


def load_cfg():
    return json.loads(CFG.read_text()) if CFG.exists() else {}


def save_cfg(c):
    CFG.write_text(json.dumps(c, indent=2))


class FSTransport:
    def __init__(self, base):
        self.base = Path(base) / "YoloStudio" / "jobs"
        self.base.mkdir(parents=True, exist_ok=True)

    def put(self, job, name, src):
        d = self.base / job; d.mkdir(exist_ok=True)
        shutil.copy(src, d / name)

    def put_json(self, job, name, obj):
        d = self.base / job; d.mkdir(exist_ok=True)
        (d / name).write_text(json.dumps(obj))

    def get_json(self, job, name):
        p = self.base / job / name
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except Exception:
            return None

    def list(self, job):
        d = self.base / job
        return [f.name for f in d.iterdir()] if d.exists() else []

    def get(self, job, name, dest):
        shutil.copy(self.base / job / name, dest)


class DriveTransport:
    SCOPES = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/userinfo.email", "openid"]

    @staticmethod
    def creds(interactive=False):
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        tok = ROOT / "token.json"
        c = Credentials.from_authorized_user_file(str(tok), DriveTransport.SCOPES) if tok.exists() else None
        if c and not c.valid and c.refresh_token:
            c.refresh(Request()); tok.write_text(c.to_json())
        if not c or not c.valid:
            if not interactive:
                raise RuntimeError("Not signed in to Google")
            cs = ROOT / "client_secret.json"
            if not cs.exists():
                raise RuntimeError("client_secret.json missing: see README (create a Google OAuth 'Desktop app' client)")
            c = InstalledAppFlow.from_client_secrets_file(str(cs), DriveTransport.SCOPES).run_local_server(port=0)
            tok.write_text(c.to_json())
        return c

    def __init__(self):
        from googleapiclient.discovery import build
        self.svc = build("drive", "v3", credentials=self.creds())
        self._dirs = {}
        jobs = self._folder("jobs", self._folder("YoloStudio", "root"))
        self.jobs = jobs
        self._jobs = {}

    def _folder(self, name, parent):
        q = f"name='{name}' and '{parent}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        r = self.svc.files().list(q=q, fields="files(id)").execute()["files"]
        if r:
            return r[0]["id"]
        return self.svc.files().create(body={"name": name, "parents": [parent], "mimeType": "application/vnd.google-apps.folder"}, fields="id").execute()["id"]

    def _job(self, job):
        if job not in self._jobs:
            self._jobs[job] = self._folder(job, self.jobs)
        return self._jobs[job]

    def _find(self, job, name):
        q = f"name='{name}' and '{self._job(job)}' in parents and trashed=false"
        r = self.svc.files().list(q=q, fields="files(id)").execute()["files"]
        return r[0]["id"] if r else None

    def put(self, job, name, src):
        from googleapiclient.http import MediaFileUpload
        m = MediaFileUpload(str(src), resumable=True)
        fid = self._find(job, name)
        if fid:
            self.svc.files().update(fileId=fid, media_body=m).execute()
        else:
            self.svc.files().create(body={"name": name, "parents": [self._job(job)]}, media_body=m).execute()

    def put_json(self, job, name, obj):
        tmp = ROOT / "projects" / ".tmp.json"
        tmp.write_text(json.dumps(obj)); self.put(job, name, tmp)

    def get(self, job, name, dest):
        from googleapiclient.http import MediaIoBaseDownload
        fid = self._find(job, name)
        with open(dest, "wb") as f:
            dl = MediaIoBaseDownload(f, self.svc.files().get_media(fileId=fid))
            done = False
            while not done:
                _, done = dl.next_chunk()

    def get_json(self, job, name):
        fid = self._find(job, name)
        if not fid:
            return None
        try:
            return json.loads(self.svc.files().get_media(fileId=fid).execute())
        except Exception:
            return None

    def email(self):
        return self.svc.about().get(fields="user(emailAddress)").execute()["user"]["emailAddress"]

    def path_folder(self, parts):
        key = "/".join(parts)
        if key not in self._dirs:
            pid = "root"
            for p in parts:
                pid = self._folder(p, pid)
            self._dirs[key] = pid
        return self._dirs[key]

    def put_path(self, parts, name, src, mime=None):
        """Upload/replace src at My Drive/<parts>/<name>; returns file id."""
        from googleapiclient.http import MediaFileUpload
        pid = self.path_folder(parts)
        q = "name='%s' and '%s' in parents and trashed=false" % (name.replace("'", "\\'"), pid)
        r = self.svc.files().list(q=q, fields="files(id)").execute()["files"]
        m = MediaFileUpload(str(src), mimetype=mime, resumable=False)
        if r:
            return self.svc.files().update(fileId=r[0]["id"], media_body=m, fields="id").execute()["id"]
        body = {"name": name, "parents": [pid]}
        if mime:
            body["mimeType"] = mime
        return self.svc.files().create(body=body, media_body=m, fields="id").execute()["id"]

    def list(self, job):
        r = self.svc.files().list(q=f"'{self._job(job)}' in parents and trashed=false", fields="files(name)").execute()["files"]
        return [f["name"] for f in r]


def get_transport():
    c = load_cfg()
    if c.get("mode") == "fs" and c.get("drive_path"):
        return FSTransport(c["drive_path"])
    if c.get("mode") == "api":
        return DriveTransport()
    raise RuntimeError("Colab link not configured. Open the 'Colab' tab and set it up.")
