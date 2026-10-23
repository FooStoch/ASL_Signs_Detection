# Computerpreter

**Web-app**: https://computerpreter.streamlit.app/

**Project Github**: https://github.com/FooStoch/ASL_Signs_Detection

**Model Training Github**: https://github.com/FooStoch/Computerpreter-Training

**Presentation**: https://docs.google.com/presentation/d/1tQ2sYysGiEH8UfrWaAOD-ggDFwBp7mRGoyiaZViza7E/edit?usp=sharing

**Full presentation and demo video (6 min)**: https://www.youtube.com/watch?v=1eQUagd-sbU&t=50s

**Paper**: https://docs.google.com/document/d/1W1qUcp0b5JO8nVbPvNwcDe_E6gxKtxKbMr_LEGp5OMQ/edit?usp=sharing

Computerpreter targets the massive communication gap facing 11 million Deaf and Hard of Hearing Americans, addressing an environment where live interpreters are costly, lipreading yields a mere 12.4% sentence accuracy, and VRS and VRI are useless for everyday conversations. To overcome the structural flaws of previous student innovations that relied on static 2D CNN image models, we engineered a free, robust Streamlit cloud web-app capable of cross-modal, real-time translation. For fingerspelling, we tracked 21 hand landmarks using MediaPipe Hands and trained a small MLP that achieved 98.84% testing accuracy over 25,000 images. For dynamic signs where motion over time alters meaning, we processed sequential video frames through MediaPipe Holistic to map 105 hand, body, and face landmarks and trained 9 different models, including Bi-LSTM, Residual TCN, and Transformer. Our final model had around 500,000 parameters and hit a final classification accuracy of over 88.1%. Finally, our app translates raw, predicted sign gloss into grammatically correct English using an online chatbot API, while integrating OpenAI's Whisper model to seamlessly handle spoken audio-to-text translation for hearing participants. After improvements and feedback from talks with students at the Jean Massieu School for the Deaf and conversations among members at the Sanderson Community Center for the Deaf and Hard of Hearing, we ultimately save communication times by up to 62% compared to traditional phone typing, and our app has over 40 regular users from the Deaf community! Recently, we secured official U.S. copyright registration for our system architecture and algorithm pipeline (U.S. Copyright Registration No. TXu 2-561-060; registered July 24, 2026) under an ARTS Act fee waiver, and currently, we are collaborating with Sorenson Communications, Inc. on data collection and improving our models.

Our Streamlit app is free on the cloud! It needs `model.p` for the random forest classifier for fingerspelling detection and `best_model.pth` for the RNN for dynamic sign recognition.

`best_model.pth` is the result of the data preprocessing and training pipeline, made of `ms-asl-wlasl-video-landmark-extractor.ipynb` and `ms-asl-wlasl-video-detection-model.ipynb`.

`asl_inference.py` is mandatory for running the code. It is what the web app is based on.
