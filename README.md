# music-transcription
Transcribe guitar recordings in WAVE format to notes and tablatures (Guitar Pro 5 format).  
Pipeline steps:
- Onset detection
- Polyphonic pitch detection
- String and fret detection
- Tempo detection
- Mapping of onset times to discrete notes in measures
- GP5 export

## Installation
This is currently only tested for Windows 10, but other platforms should work as well.
1. Download and extract this repository.
2. Download and install Anaconda3 64-bit if you don't already have it.
3. Create and activate a new environment containing all relevant modules using the following commands:
   ```
   conda env create -f $INSTALLDIR/conda_env.yml
   activate music_transcription
   ```
4. Change the following Keras attributes in the file %USERPROFILE%/.keras/keras.json ($HOME/.keras/keras.json for \*nix):
   ```
   "image_data_format": "channels_first"
   "backend": "theano"
   ```
   More info: https://keras.io/backend/  
5. (optional) To speed up the CNNs used for onset and pitch detection and if you have a fast GPU, consider running Keras / Theano on the   GPU. For even more speed, activate CuDNN and CNMeM. See these two links:  
   http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/  
   http://ankivil.com/making-theano-faster-with-cudnn-and-cnmem-on-windows-10/  
   
   This is recommended if you plan to train new models.

## Getting started
Open an Anaconda Prompt and switch the working directory to $INSTALLDIR/pipelines.  
  
To transcribe a recording using polyphonic pitch detection:
```
python guitar_pipeline.py ..\example_recordings\instrumental_lead.wav
```
  
To transcribe a recording using monophonic pitch detection and a custom output path:
```
python guitar_pipeline.py ..\example_recordings\instrumental_lead.wav -m mono -p instrumental_lead.mono.gp5
```
  
To compare two GP5 files:
```
python compare_gp5.py instrumental_lead.gp5 instrumental_lead.mono.gp5 --outfile instrumental_lead_poly_vs_mono.gp5
```
The output is another GP5 file with three different tracks, one for common notes and two more describing the notes in the differing regions for each file.  
  
To see what other options guitar_pipeline.py has to offer:
```
python guitar_pipeline.py -h
```
To see what other options compare_gp5.py has to offer:
```
python compare_gp5.py -h
```
