import os

import pytest


@pytest.fixture(scope="session")
def spark():
    pytest.importorskip("pyspark")
    from ridepay.etl import spark_session

    os.environ.setdefault("PYSPARK_PYTHON", "python")
    s = spark_session("tests", memory="1g")
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()
