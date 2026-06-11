"""Thin wrapper over the MinIO (S3-compatible) client for storing uploaded images.

Background — what is MinIO?
    MinIO is an open-source object storage server whose API is wire-compatible with Amazon S3. An
    "object" is any blob of bytes (image, PDF, model weights …) identified by two coordinates:

        bucket   — a named top-level container, loosely like a filesystem volume.
                   Convention in this project: one bucket per environment ("htr-uploads").
        key      — the path-like string within the bucket, e.g. "uploads/<uuid>.png".
                   Keys contain slashes for organisational clarity but the storage is flat
                   (there are no real directories — slashes are just part of the string).

    The combination (bucket, key) is globally unique within a MinIO cluster and is what we persist
    in the Postgres `upload_result` table so we can later retrieve or generate a URL for the image.

Background — why a wrapper class?
    The real `minio.Minio` client cannot be called in unit tests without a running server. By
    accepting `client` as a constructor argument we allow tests to inject a `FakeMinio` that
    records calls without any network traffic. `from_config()` is the only place that imports
    `minio` and constructs the real client — making that import lazy means the module can be
    imported (and tested) even when the `minio` package is not installed.

Background — presigned URLs:
    Rather than proxying image bytes through the API server, we use MinIO's presigned-GET feature.
    A presigned URL is a time-limited URL that encodes an HMAC signature in the query string. The
    browser can fetch it directly from MinIO — reducing API server bandwidth and keeping the API
    responses small (just the URL string, not megabytes of image data).
    The default expiry of 3600 s (1 hour) is sufficient for a single dashboard session; the URL
    becomes invalid after that and a new one must be generated.

Minio package import note:
    The `minio` package is imported lazily inside `from_config` so importing this module never
    requires the dependency. Tests inject a fake client directly via __init__.
"""
from __future__ import annotations

import io
import os
import uuid
from datetime import timedelta

from htr_sp5 import config

# ---------------------------------------------------------------------------
# Extension normalisation map
#
# Web browsers upload images with various extension capitalisations (.PNG, .JPG, .JPEG …).
# We normalise to lower-case and collapse .jpeg → .jpg so object keys are predictable.
# Any extension not in this map (e.g. .bmp, .tiff) falls back to .png — a safe assumption
# for an HTR system that expects raster images.
#
# Why normalise?
#   Content-type inference at download time can depend on the file extension when a browser
#   renders an object URL. Consistent extensions avoid ambiguity, and the whitelist prevents
#   arbitrary file types being stored (defence in depth before the API layer validates MIME).
# ---------------------------------------------------------------------------
_ALLOWED_EXT: dict[str, str] = {
    ".png": ".png",
    ".jpg": ".jpg",
    ".jpeg": ".jpg",   # .jpeg and .jpg are the same format; we store everything as .jpg
}


class MinioObjectStore:
    """Thin façade over the MinIO (or any S3-compatible) client.

    Responsibilities:
        1. Generate collision-proof object keys (`new_object_key`).
        2. Lazily create the bucket on first upload (`_ensure_bucket`).
        3. Upload raw bytes and return the stored key (`put_object`).
        4. Return a presigned GET URL for a previously stored key (`presigned_get_url`).

    The constructor takes a `client` argument rather than constructing one internally so the
    class is fully testable without a real MinIO server (see FakeMinio in the test file).
    Production code should use `from_config()` which builds the real client from environment
    variables defined in `htr_sp5.config`.
    """

    def __init__(self, client, bucket: str) -> None:
        """
        Args:
            client: An object with the same method signatures as `minio.Minio`.
                    In tests this is FakeMinio; in production it is a real Minio instance.
            bucket: Name of the MinIO bucket to use (e.g. "htr-uploads"). The bucket will be
                    created automatically if it does not exist (see `_ensure_bucket`).
        """
        self._client = client
        # Bucket name stored once; all operations in this instance target this single bucket.
        # Keeping it in one attribute means changing the bucket for a deployment only requires
        # updating the HTR_MINIO_BUCKET env var — no code changes needed.
        self._bucket = bucket

    # ------------------------------------------------------------------
    # Construction from environment / config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls) -> "MinioObjectStore":
        """Build a production MinioObjectStore from environment variables.

        The `minio` package is imported HERE (and nowhere else) so that the entire module — and
        any code that imports it — can be loaded without `minio` being installed. The test suite
        uses `MinioObjectStore(client=FakeMinio(), bucket=…)` and never touches this method.

        Config variables used (all defined in htr_sp5.config):
            MINIO_ENDPOINT   — host:port, e.g. "localhost:9000"
            MINIO_ACCESS_KEY — username / access key ID
            MINIO_SECRET_KEY — password / secret access key
            MINIO_SECURE     — True → HTTPS, False → HTTP (local dev)
            MINIO_BUCKET     — bucket name, default "htr-uploads"

        Returns:
            A fully configured MinioObjectStore ready to accept uploads.
        """
        # Lazy import: only executed when from_config() is actually called (never in tests).
        from minio import Minio  # type: ignore[import-untyped]  # minio stubs not published

        client = Minio(
            config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            # secure=True means TLS (HTTPS). Required for any non-localhost deployment so
            # credentials are not transmitted in plaintext.
            secure=config.MINIO_SECURE,
        )
        return cls(client=client, bucket=config.MINIO_BUCKET)

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def new_object_key(self, filename: str) -> str:
        """Generate a collision-proof object key for an uploaded file.

        The key format is:  uploads/<uuid_hex><normalised_ext>
        e.g.               uploads/3f2e1a0b…c9d8e7f6.png

        Why UUID-based keys?
            Using the original filename as the key would cause silent overwrites if two users
            upload files with the same name. A UUID hex string (32 hex characters = 128 bits of
            randomness) makes collisions astronomically unlikely and is safe for URL paths.

        Why a fixed "uploads/" prefix?
            MinIO supports lifecycle rules and IAM policies scoped to key prefixes. Having all
            user-uploaded images under "uploads/" makes it straightforward to, e.g., apply an
            object-expiry rule only to that prefix or grant read-only access to just those keys.

        Args:
            filename: The original filename supplied by the browser (e.g. "My Photo.PNG").
                      Only the extension is used; the stem is discarded.

        Returns:
            A string like "uploads/3f2e1a0b…c9d8e7f6.png" suitable as a MinIO object key.
        """
        # os.path.splitext handles edge cases: "photo.PNG" → (".PNG"), ".hidden" → ("", ".hidden")
        raw_ext = os.path.splitext(filename)[1].lower()  # normalise capitalisation first
        # Fallback to .png for unknown extensions — safe for HTR which expects raster images.
        ext = _ALLOWED_EXT.get(raw_ext, ".png")
        # uuid4() is random (v4) rather than time-based (v1) to avoid leaking upload timestamps
        # in the key, and .hex gives us a compact 32-character string without hyphens.
        return f"uploads/{uuid.uuid4().hex}{ext}"

    # ------------------------------------------------------------------
    # Bucket management
    # ------------------------------------------------------------------

    def _ensure_bucket(self) -> None:
        """Create the target bucket if it does not already exist (idempotent).

        MinIO raises an error if you call put_object on a bucket that does not exist. By calling
        this method before every upload we make the first upload self-configuring — the operator
        does not need to manually create the bucket via the MinIO console or CLI.

        Why not create once at startup?
            Checking at startup would fail if MinIO is unreachable at boot time (e.g. it starts
            up more slowly than the API server). Deferring the check to the first actual upload
            makes the service more resilient to transient MinIO unavailability on startup.

        The check is a single HEAD-like request (`bucket_exists`). After the first successful
        upload, subsequent calls exit on the `if` branch without making a `make_bucket` call.
        """
        if not self._client.bucket_exists(self._bucket):
            # We use an explicit check-then-create pattern (bucket_exists → make_bucket) rather
            # than relying on make_bucket being idempotent.  Idempotency behaviour of make_bucket
            # has varied across minio-py versions: some silently absorb the "BucketAlreadyExists"
            # error, others re-raise it.  The explicit guard is reliable regardless of version.
            #
            # Trade-off: this is NOT safe under concurrent bucket-creation (two writers could both
            # see _exists=False and then both call make_bucket).  That race is acceptable here
            # because this is a single-writer thesis prototype — one API server process, one bucket.
            # A production system would use a server-side "create if not exists" primitive or a
            # distributed lock instead.
            self._client.make_bucket(self._bucket)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def put_object(
        self,
        object_key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload raw bytes to MinIO under the given key, creating the bucket if needed.

        The method is intentionally simple: it wraps bytes in a BytesIO stream (which minio-py
        requires for streaming uploads), calls the underlying client, and returns the key so the
        caller can persist it to the DB in one step:

            key = store.put_object(store.new_object_key(filename), raw_bytes, "image/png")
            db.insert_upload(key, …)

        Why accept bytes rather than a file-like object?
            FastAPI's UploadFile.read() returns bytes and it is already in memory at that point.
            Accepting bytes keeps the signature simple and avoids confusion about stream position
            (a caller might forget to seek(0) before passing a file object).

        Why return the key?
            Allows one-liner chaining with new_object_key() and keeps the calling code minimal.

        Args:
            object_key:   The MinIO object key, typically produced by `new_object_key()`.
            data:         Raw image bytes to store.
            content_type: MIME type for the stored object. MinIO records this and returns it in
                          the Content-Type header when the object is fetched via a presigned URL,
                          which is important for correct browser rendering of the image.

        Returns:
            The object_key that was stored (same value passed in, for convenient chaining).
        """
        # Lazily create the bucket on first upload (no-op on subsequent uploads).
        self._ensure_bucket()

        # minio-py's put_object requires a file-like object with a .read() method, not plain
        # bytes. io.BytesIO wraps the bytes buffer without copying; the `length` argument tells
        # the server exactly how many bytes to expect (required for S3-compatible APIs that cannot
        # use chunked Transfer-Encoding by default).
        self._client.put_object(
            self._bucket,
            object_key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

        # Return the key so the caller can store it in the DB without a separate variable.
        return object_key

    # ------------------------------------------------------------------
    # Presigned URL generation
    # ------------------------------------------------------------------

    def presigned_get_url(self, object_key: str, expires_seconds: int = 3600) -> str:
        """Return a time-limited GET URL the browser can use to load the image directly.

        How presigned URLs work:
            The MinIO server signs the URL with an HMAC of (method, bucket, key, expiry, …)
            using the secret key. The browser sends the URL to MinIO directly — bypassing the
            API server entirely. MinIO verifies the signature and the expiry timestamp before
            serving the object. If the URL has expired or the signature is invalid, MinIO returns
            403 Forbidden.

        Why bypass the API server?
            The API server would otherwise need to proxy potentially large image files, consuming
            memory and bandwidth. Presigned URLs let MinIO handle the heavy lifting while the API
            server only has to generate a short string. This is the standard pattern in S3-based
            architectures (see AWS S3 presigned URL documentation for the canonical reference).

        Why 1-hour default?
            One hour is long enough to cover a full dashboard session (the user opens the page,
            browses history, the images load). After that the URLs are regenerated on the next
            page load. Using a shorter TTL (e.g. 5 minutes) would cause images to 403 mid-session
            on slow connections; using a longer TTL increases the risk window if a URL leaks.

        Args:
            object_key:      The stored key (e.g. "uploads/<uuid>.png").
            expires_seconds: How many seconds until the URL becomes invalid. Default 3600 (1 h).

        Returns:
            A URL string the browser can GET directly from MinIO for the duration of `expires`.
        """
        # timedelta is passed as the `expires` keyword so the minio-py client converts it to the
        # correct query-string parameters (X-Amz-Expires for v4 signature). Passing an integer
        # directly would raise a TypeError in minio-py >= 7.x (it expects a timedelta, not int).
        return self._client.presigned_get_object(
            self._bucket,
            object_key,
            expires=timedelta(seconds=expires_seconds),
        )
