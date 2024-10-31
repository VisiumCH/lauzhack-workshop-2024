# Introduction to RAG workshop

## Intro

This workshop will introduce how Retrieval-Augmented Generation (RAG) works and how to set up a RAG system on your own device using Ollama, LlamaIndex, and Chroma DB. You’ll explore how RAG improves AI-generated responses by retrieving relevant information from a vector database. We’ll guide you through installing and configuring the necessary tools and demonstrate how to store and query your data. By the end, you’ll be able to efficiently retrieve and generate answers based on your local documents!



## What is a RAG?

TODO rag pattern diagram

<!-- Remember we need slides to back this-->


## Table of contents


| Content    | Time estimate | Description 
| -------- | ------- | ------- |
|     Exercise 0     | xx minutes   | ... |
|     Exercise 1     | xx minutes   | ... |
|     Exercise 2     | xx minutes   | ... |
|     Exercise 3     | xx minutes   | ... |
|     Exercise 4     | xx minutes   | ... |
|     Exercise 5     | xx minutes   | ... |

## Hardware requirements

Currently only tested on Mac M1
TODO: (@unesmu) test on Windows ?

## Exercises

TODO: Make the dropdown summary text much larger
<details>
<summary> <b>Exercise 0 : Setting up your python environment </b> </summary>
<br>

What is a virtual environment you ask ?

A virtual environment is a sandbox that will contain all the python packages you need for a given project. Each virtual environment will have it's own copy of packages. 

One concrete example of why it's a good idea to use virtual environments: 
> It's late, you're scrambling to finish an ML project you have to return soon but realise you need to install a new package. You do that. The next day you work on your semester project and realise nothing runs because of some strange import error. Now you are sad :(

We want YOU to be happy, so we'll show you how to install a virtual environment :)

TODO: Put the world if everyone used virtual environments meme here 


<!-- Let's keep it simple, just use venv and the requirements txt file, can give some alternatives like conda--- we can quickly check what's on the cs443 website aand ada website too see what they still recommend -->
There are many ways you can create a virtual environment, from simple to more complexe:
- using venv
- using anaconda ( or conda )
- using pipenv, poety or uv



We usually recommend using pipenv, poerty or uv, but for simplicity we'll use anaconda. 

For simplicity we'll show you how to use venv and conda, you only need to do with one of the methods: 

<details>


<summary>Using venv </summary>
<br>
TODO
</details>


<details>
<summary>Using conda </summary>
<br>

Install anaconda
```
```

Create a virtual environment, in a terminal window run:

```
conda create env --name lauzhackviz python=3.11
```

Install the required python  packages using pip
```
pip install -r requirements.txt
```

</details>




</details>

--- 

<details>
<summary><b> Exercise 1 : Installing ollama and downloading the embedding and llm models</b> </summary>
<br>
TODO
</details>


--- 
<details>
<summary><b> Exercise 2 : Creating your index, trying out retrieval</b> </summary>
<br>
TODO
</details>

--- 
<details>
<summary><b> Exercise 3 : Testing the LLM</b> </summary>
<br>
TODO
</details>

--- 
<details>
<summary><b> Exercise 4 : Joining the two components to make your RAG</b> </summary>
<br>
TODO

</details>

--- 
<details>
<summary><b> (bonus) Exercise 5 : Using ... to convert pdf to txt </b> </summary>
<br>
TODO

<!-- If we don't have time to do this, we can put the "The proof is left as an exercise to the reader " meme -->


</details>

--- 

<details>
<summary><b>(bonus) Exercise 6 : Using openAI models instead </b></summary>
<br>
TODO
<!-- If we don't have time to do this, we can put the "The proof is left as an exercise to the reader " meme -->
</details>