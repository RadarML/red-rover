Data Processing Pipeline
========================

.. image:: processing.svg
   :alt: Data processing pipeline


Scripts
-------

Scripts are categorized as follows:

- **Convert**: convert data format for compatibility with external software.
- **Create**: generate a specific representation based on the input data.
- **Export**: convert or copy data for training or other distribution.
- **Get**: print or visualize dataset metadata.
- **Render**: create a video visualization of the data.
- **Run**: a compute (GPU) intensive operation to apply a given algorithm.

.. toctree::
   :maxdepth: 2

   scripts/convert.rst
   scripts/create.rst
   scripts/export.rst
   scripts/get.rst
   scripts/render.rst
   scripts/run.rst
