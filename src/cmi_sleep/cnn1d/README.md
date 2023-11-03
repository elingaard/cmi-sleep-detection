# 1D-CNN Approach

## Data preprocessing

The original dataset is very big, i.e. 100 million rows. The following preprocessing has been performed to reduce memory consumption and model training time:

- Use `pyarrow` backend in pandas for efficient data types and faster operations
- Time series is resampled from 5sec to 1min frequency
- Any segments where no events have been annotated for 24 hours are discarded

## Data sampling

During training a random segment of 12 hours is extracted from each unique `series_id`. 

## Model

The overall objective of the model is to predict whether a child is asleep or not at a given timestamp. These binary predictions can then be postprocessed to yield onset and wakeup times by finding peaks and valleys in the signal.

The 1D CNN model consists of three main components:
- A feature pyramid network is used for initial feature generation. Here 3 different convolutional layers with different dilations factor are used to extract features at different time scales.
- A series of 1D ResNet blocks are used for feature processing.
- A classification head that uses 1D convolution to project the generated *N* dimensional feature signal to a binary prediction for each time step

Weight decay and dropout layer(s) are used to prevent overfitting.

## Post-processing

Given a binary array of predictions indication whether the child is asleep or not the following postprocessing is performed

- Smooth the signal in 30min windows using a Savitzky-Golay filter
- Binary closing operation to close any gaps smaller than 30mins
- Find plateaus in the signal using `scipy.signal.find_peaks`, start and end indices of the plateus indicate onset and wake up times.
- Discard any plateaus smaller than 2 hours

## Future improvements

- Use a more appropriate padding type than zero-padding
- Use stochastic weight averaging
- Think of alternative objectives, current binary target of being asleep or not is not necesarrily the most appropriate for event detection  