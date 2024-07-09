This project showcases how to create a searchable database from a Bilibili video transcript, perform similarity search queries using the FAISS library, and respond to user questions with relevant and precise information. 
It makes a call to the glm-4 model using ZhipuAI API, so you will need a zhupuai api key.

demo see below:
![Gradio demo](Gradio_demo1.jpg)

# Creating a Python environment

To create a Python environment, you can use either `venv` or `conda`. Here are the steps for each option:

Using `venv`:

```shell
cd your-project-folder
python3 -m venv env
source env/bin/activate
```

Using `conda`:

```shell
cd your-project-folder
conda create -n demo-env python=3.10
conda activate demo-env
```
​    

# Installing the required dependencies

To install the required dependencies, run the following command:

```shell
pip install -r requirements.txt
```





# Setting up the keys in a .env file

To set up the keys in a `.env` file, follow these steps:

1. Create a `.env` file in the root directory of your project.
2. Inside the file, add your ZHIPUAI API key and other required keys:

```shell
ZHIPUAI_API_KEY="your_api_key_here"
SESSDATA="your_key_here"
BILI_JCT="your_key_here"
BUVID3="your_key_here"
```

For detailed instructions on obtaining these credentials to access bilibili, refer to the guide [here](https://nemo2011.github.io/bilibili-api/#/get-credential).

3. Save the file and close it.

In your Python script or Jupyter notebook, load the `.env` file using the following code:

```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
```

