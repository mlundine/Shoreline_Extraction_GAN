# Shoreline_Extraction_GAN

Generating binary land vs. water masks from coastal RGB imagery using pix2pix and cycle-GAN.

This repo will contain the following:
1. Code to help with setting up the training data for either of these image-to-image translation GANs.
2. Code to train both models.
3. Code to run the models.
4. Code to process the results and extract shorelines into GIS format.
5. Trained models for Delmarva's coast.

![cape_ex](/images/capehenlopen_ex.png)

![iri_ex](/images/iri_example.png)

![cape_map](/images/capeHenlopen_length_years.png)

![more_examples](/images/input_output_shoreline.png)


# Downloading satellite data

CoastSat allows users to download Landsat 5, 7, 8 and Sentinel-2 imagery from anywhere around the world.
CoastSat also includes a number of useful preprocessing tools.
Data for this project was downloaded using CoastSat.

# Labelling data.

I used labelme to manually segment imagery into two classes: land and water.

# Converting jsons to pngs

# Building a pix2pix and/or cycle-GAN training set.

# Training

# Deployment

# Processing Results


