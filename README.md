# AutomatedEssayGrading
Automated essay grader using logistic regression to classify "good" writing.

Basic Structure of the Code:
1. All imports of the modules and libraries are located at the top of the code.
2. Conversion and reading of downloaded files follow, specifically the data set downloaded from Kaggle and the corpus set. We also utilized nltk’s English corpus and converted it into a list for future word comparison.
3. Next, we pre-processed the data by establishing stop words and setting up tokenizers and lemmatizers. We also included our own file called contractions.py that has a dictionary of all the contractions.
4. After pre-processing, our methods are located, specifically 3 methods: writer_label() which classifies all of the data, spelling_errors() which check for spelling mistakes, and sophisticated() which check for the number of sophisticated words.
5. Then, we proceed to transfer the data from our Kaggle dataset into a dataframe that we created.
6. After creating a dataframe, we start to create our features, including essay length, number of unique words, number of sentences, average sentence length, spelling errors, number of sophisticated words, and number of grammar errors.
7. We converted all the features into numpy arrays and added the rest of these features into the dataframe.
8. The following sections are all scatter plots and correlation coefficients that show the relation between each feature and the label.
9. The final portion includes building the logistic regression model, and testing our model’s performance. 
10.  After running the code, the classification report will be printed which will show the precision, recall, and f-measure scores for the following classes: 0 and 1.

Compliling the Code
1. Download EssayGrading.py, contractions.py, and requirements.txt and place into the same project folder.
2. Go to python virtual environment (ven) and type “pip install -r requirements.txt”. This will import all the modules and libraries that we used for our project.
3. Download training_set_rel3.tsv and BigCorpus_5000.cvs and place both into your home directory. This is the data set that we used from an online source.
4. Comments are noted throughout the entire code that explain what each method does, or what the section of code will return.
