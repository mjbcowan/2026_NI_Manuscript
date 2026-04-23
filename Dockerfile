FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

LABEL description="2026 NI Manuscript CellDIVE spatial analysis environment"

# Install mamba for faster conda solves
RUN conda install -n base -c conda-forge mamba -y

WORKDIR /workspace

# Copy environment spec first so Docker can cache the layer
COPY environment.yml .

# Create the project environment
RUN mamba env create -f environment.yml && conda clean --all -f -y

# Make the conda env's Python the default
ENV PATH="/opt/conda/envs/ni_manuscript/bin:$PATH"

# Copy the project code
COPY . .

# Default command: launch a shell inside the environment
CMD ["/bin/bash"]
