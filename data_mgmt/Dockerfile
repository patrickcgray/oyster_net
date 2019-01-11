# We will use the Anaconda customized verison of Ubuntu 16.04 for our image
FROM continuumio/anaconda3

# Set the working directory to the current directory
WORKDIR .

# Updating Ubuntu packages
#RUN apt-get update && yes|apt-get upgrade

# setup all python packages with conda
RUN conda create -n geo_env -c conda-forge rasterio matplotlib folium fiona pyproj shapely notebook pandas cartopy scikit-learn

# switch this to a YAML file to specify specific versions of things
#ADD environment.yml /tmp/environment.yml
#RUN conda env create -f /tmp/environment.yml

# start the virtual environment
RUN echo "source activate geo_env" > ~/.bashrc
ENV PATH /opt/conda/envs/geo_env/bin:$PATH 

RUN pip install pypng

# Configuring access to Jupyter
RUN mkdir /opt/notebooks
RUN jupyter notebook --generate-config --allow-root
# set the password as 'root'
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py

# Jupyter listens port: 8888
EXPOSE 8888

# Run Jupyter notebook as Docker main process
CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/host", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

# command to run if you don't want the above to run by default
# jupyter notebook --allow-root --notebook-dir=/host --ip=0.0.0.0 --port=8888 --no-browser

# command to run to build this image when in the same directory as this dockerfile
# docker build -t geo .

# command to run to setup this container from a built image
# use a different port than normal (8889 instead of 8888) because we might have the other training docker container running on that 8888
# docker run --name geocont -p 8889:8888 -v ~/:/host -it geo