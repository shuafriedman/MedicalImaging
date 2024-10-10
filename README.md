# MedicalImaging

This repository contains code and resources for training, evaluating, and inferring medical imaging models. The project is structured to facilitate easy experimentation and reproducibility.

## Project Structure

- [`config.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fsfriedman%2FMedicalImaging%2Fconfig.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%222bc3c7aa-29f8-4417-b94a-a069cc9440d7%22%5D "/Users/sfriedman/MedicalImaging/config.py"): Configuration file for training and inference parameters.

- [`inference.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fsfriedman%2FMedicalImaging%2Finference.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%222bc3c7aa-29f8-4417-b94a-a069cc9440d7%22%5D "/Users/sfriedman/MedicalImaging/inference.py"): Script for running inference using 
- [`model.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fsfriedman%2FMedicalImaging%2Fmodel.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%222bc3c7aa-29f8-4417-b94a-a069cc9440d7%22%5D "/Users/sfriedman/MedicalImaging/model.py"): Script defining the model architectures.
- [`train.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fsfriedman%2FMedicalImaging%2Ftrain.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%222bc3c7aa-29f8-4417-b94a-a069cc9440d7%22%5D "/Users/sfriedman/MedicalImaging/train.py"): Script for training the model.
- [`upload.ipynb`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fsfriedman%2FMedicalImaging%2Fupload.ipynb%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%222bc3c7aa-29f8-4417-b94a-a069cc9440d7%22%5D "/Users/sfriedman/MedicalImaging/upload.ipynb"): Jupyter notebook for uploading models to github.
- [`utils.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fsfriedman%2FMedicalImaging%2Futils.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%222bc3c7aa-29f8-4417-b94a-a069cc9440d7%22%5D "/Users/sfriedman/MedicalImaging/utils.py"): Utility functions used across the project.
- [`all_results.csv`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fsfriedman%2FMedicalImaging%2Fall_results.csv%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%222bc3c7aa-29f8-4417-b94a-a069cc9440d7%22%5D "/Users/sfriedman/MedicalImaging/all_results.csv"): CSV file containing training and evaluation results.
## Getting Started

To train the model, upload images, change the training path, and run:
```sh
python train.py
```
The training results will be saved in all_results.csv.

#### Evaluation

To run inference using a trained model, use:
```sh
python inference.py
```
