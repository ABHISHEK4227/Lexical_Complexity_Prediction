# CS60075-Team-10-Task-1

This repository contains the code for the SemEval Shared Task 1 i.e. Lexical Complexity Prediction.

**Following are the library Requirements for our code :**
  <ul>  
  <li>keras</li>
  <li>transformers</li>
  <li>textstat</li>
  <li>bert</li>
  <li>spacy </li>
  <li>sklearn</li>
  <li>numpy</li>
  <li>pandas</li>
  <li>nltk</li>
  </ul>
  
Apart from the above libraries , we also require the **glove embeddings of 42 billion** tokens having a vocabulary size of **1.9M**
<br><br>


For running the model for single word complexity run the **example_single.ipynb** file. 

For running the model for multi word complexity run the **example_multi.ipynb** file.

<b>Note:</b> <ul>
  <li>Above example ipynb files are made for google colab.. If you are using those thn it will download the required libraries and GloveEmbedding file itself. Other wise please make sure to satisfy the library dependency and to download and place the glove embedding file of the required config [glove.42B.300d.txt] as mentioned above adjacent to the files.</li>
  <li>In MultiWord prediction we were able to improve the pearson score a bit on the day of the deadline but the video was recorded a day before that so there is a bit mismatch in pearson number of the MultiWord Prediction shown in the video and the one we submitted. The final results is submitted as screenshot in the report.</li>
  </ul>


The following ipynb files contains the results we obtained :
<ul>
  <li>For Single word complexity prediction:  <b>LCP_SingleWordPrediction.ipynb</b></li>
  <li>For Multi word complexity  prediction:  <b>LCP_MultiWordPrediction.ipynb</b></li>
 </ul>












