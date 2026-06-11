"""MinioObjectStore wraps minio-py. We inject a fake client so no real MinIO is needed.

All tests use FakeMinio — a hand-rolled test double that records every call made to it.
No real MinIO server is required; the minio package does not even need to be installed
(MinioObjectStore.from_config is the only place that imports `minio` and we never call it here).
"""
import datetime
import io
from htr_sp5.objectstore import MinioObjectStore


class FakeMinio:
    """Minimal test double for the minio.Minio client.

    Records every call so tests can assert on exact arguments without any network traffic.

    Attributes:
        put_calls:     list of (bucket, key, data_bytes, length, content_type) tuples —
                       one entry per put_object call.
        made_buckets:  list of bucket names passed to make_bucket — length tells us how
                       many times the bucket-creation path was taken.
        presigned_expires: list of `expires` kwargs passed to presigned_get_object — used
                           to assert the caller passes a datetime.timedelta, not a bare int.
        _exists:       controls what bucket_exists returns; starts False (bucket absent),
                       flips to True after the first make_bucket call.
    """

    def __init__(self):
        self.put_calls = []
        self.made_buckets = []
        # Record the `expires` argument of every presigned_get_object call so tests can
        # verify that a timedelta (not a bare integer) is passed — minio-py >= 7 requires this.
        self.presigned_expires = []
        self._exists = False

    def bucket_exists(self, bucket):
        # Returns the current existence state — tests can preset this to True to skip creation.
        return self._exists

    def make_bucket(self, bucket):
        # Appending to made_buckets lets tests count how many times creation was attempted.
        self.made_buckets.append(bucket)
        # Flip _exists so that subsequent bucket_exists calls return True (simulating a real
        # server where a bucket cannot be created twice).
        self._exists = True

    def put_object(self, bucket, key, data, length, content_type=None):
        # data is a BytesIO; .read() consumes it now so the recorded bytes are a plain bytes
        # object that tests can assert against without worrying about stream position.
        self.put_calls.append((bucket, key, data.read(), length, content_type))

    def presigned_get_object(self, bucket, key, expires=None):
        # Record the expires value so tests can assert on its type (must be timedelta).
        self.presigned_expires.append(expires)
        return f"http://minio/{bucket}/{key}?sig=abc"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_put_object_creates_bucket_then_uploads():
    """First put_object on a missing bucket: bucket is created then object uploaded."""
    fake = FakeMinio()
    store = MinioObjectStore(client=fake, bucket="htr-uploads")
    key = store.put_object("uploads/a.png", b"PNGBYTES", content_type="image/png")
    assert key == "uploads/a.png"
    assert fake.made_buckets == ["htr-uploads"]          # bucket didn't exist -> created once
    assert fake.put_calls[0][0] == "htr-uploads"
    assert fake.put_calls[0][2] == b"PNGBYTES"
    assert fake.put_calls[0][3] == len(b"PNGBYTES")


def test_put_object_second_call_does_not_recreate_bucket():
    """Second put_object must NOT call make_bucket again — bucket already exists after first upload.

    The _ensure_bucket guard (bucket_exists → make_bucket) should be a no-op on the second
    call because make_bucket sets _exists=True on the FakeMinio, so bucket_exists returns True
    and the make_bucket branch is skipped.  This verifies the guard works as documented.
    """
    fake = FakeMinio()  # _exists starts False — bucket absent
    store = MinioObjectStore(client=fake, bucket="htr-uploads")

    # First upload — bucket does not exist yet, so make_bucket should be called once.
    store.put_object("uploads/first.png", b"AAA", content_type="image/png")
    # Second upload — bucket now exists (_exists flipped to True by make_bucket).
    store.put_object("uploads/second.png", b"BBB", content_type="image/png")

    # make_bucket should have been called exactly once despite two uploads.
    assert fake.made_buckets == ["htr-uploads"], (
        f"Expected make_bucket called once, got: {fake.made_buckets}"
    )
    # Both objects should have been uploaded.
    assert len(fake.put_calls) == 2


def test_presigned_url_delegates_to_client():
    """presigned_get_url returns the URL produced by the underlying client."""
    fake = FakeMinio()
    fake._exists = True   # bucket already exists — skip creation
    store = MinioObjectStore(client=fake, bucket="htr-uploads")
    url = store.presigned_get_url("uploads/a.png")
    assert url == "http://minio/htr-uploads/uploads/a.png?sig=abc"


def test_presigned_url_passes_timedelta_expires():
    """presigned_get_url must pass a datetime.timedelta (not a bare int) to the client.

    minio-py >= 7.x requires a timedelta for the `expires` parameter; passing an integer
    raises TypeError at runtime.  This test asserts that MinioObjectStore correctly converts
    the expires_seconds integer into a timedelta before delegating to the client.
    """
    fake = FakeMinio()
    fake._exists = True
    store = MinioObjectStore(client=fake, bucket="htr-uploads")
    store.presigned_get_url("uploads/a.png", expires_seconds=7200)

    # The client must have received a timedelta, not a bare int.
    assert len(fake.presigned_expires) == 1, "Expected exactly one presigned_get_object call"
    received = fake.presigned_expires[0]
    assert isinstance(received, datetime.timedelta), (
        f"Expected datetime.timedelta, got {type(received).__name__}: {received!r}"
    )
    # Also verify the duration matches the requested 7200 seconds.
    assert received == datetime.timedelta(seconds=7200)


def test_new_object_key_has_uploads_prefix_and_extension():
    """new_object_key produces a well-formed key with the correct normalised extension."""
    store = MinioObjectStore(client=FakeMinio(), bucket="b")
    key = store.new_object_key("My Photo.PNG")
    assert key.startswith("uploads/") and key.endswith(".png")


def test_new_object_key_jpeg_normalises_to_jpg():
    """.jpeg (and .JPEG) must normalise to .jpg — both formats are identical, one extension wins.

    The _ALLOWED_EXT map maps ".jpeg" → ".jpg".  This test verifies that an upper-case
    .JPEG extension is first lowered and then collapsed to .jpg, not stored as .jpeg.
    """
    store = MinioObjectStore(client=FakeMinio(), bucket="b")
    key = store.new_object_key("photo.JPEG")
    assert key.endswith(".jpg"), (
        f"Expected key ending in .jpg (jpeg → jpg normalisation), got: {key!r}"
    )
