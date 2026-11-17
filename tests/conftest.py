"""Keep test runs from writing into the real flag store."""
import os
import tempfile

os.environ.setdefault("NSCA_FLAG_DIR", tempfile.mkdtemp(prefix="nsca_flags_test_"))
