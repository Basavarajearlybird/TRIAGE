"""Force every model onto the mock provider before app import, so the suite runs
offline with no keys and no network."""
import os

os.environ["TRIAGE_FORCE_MOCK"] = "1"
