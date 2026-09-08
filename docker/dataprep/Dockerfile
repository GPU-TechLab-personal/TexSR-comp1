# Data preparation - MatSynth parquet -> per-material map folders.
#
# Deliberately a THIRD image. matup and deeptex each pin a published method's
# environment and must not drift for data-wrangling reasons; pyarrow has no
# business in either. No CUDA here - this stage is pure I/O and resizing.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Same contract as the other two images: fail the BUILD, not the first run.
RUN python -c "\
import pyarrow, cv2, numpy, PIL; \
print('dataprep deps OK | pyarrow', pyarrow.__version__, '| cv2', cv2.__version__, '| numpy', numpy.__version__)"

WORKDIR /workspace
CMD ["bash"]
