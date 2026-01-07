# Computerpreter

<b>To access our cloud web app, here is the link: <a href="https://computerpreter.streamlit.app" target="_blank">https://computerpreter.streamlit.app/</a></b>

This Streamlit app is cloud-only! It needs `model.p` for the random forest classifier for fingerspelling detection and `best_model.pth` for the RNN for dynamic sign recognition.

`best_model.pth` is the result of the data preprocessing and training pipeline, made of `ms-asl-wlasl-video-landmark-extractor.ipynb` and `ms-asl-wlasl-video-detection-model.ipynb`.

`asl_inference.py` is mandatory for running the code. It is what the web app is based on.

YAY!
