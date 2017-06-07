# User Guide for window users

## 1. Install Anaconda
	from https://www.continuum.io/downloads

## 2. Install required python moduls
	Change your working directory to sentiment_analysis folder which contains requirement.txt
	> pip install -r requirement.txt

## 3. Make sure your local machine has install 64bit java

## 4. Download stanford corenlp 
	http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip

## 5. Start running stanford corenlp localhost server on your machine
	Unzip the stanford corenlp
	
	Change your working directory to the stanford-corenlp-full-2016-10-31 folder
	> cd stanford-corenlp-full-2016-10-31
	
	Make sure the current directory is the folder that contains all the *.jar files
	
	Run the localhost server on port 9000
	> java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 50000
	
## 6. Download required nltk packages
	> ipython
	> import nltk
	> nltk.download()
	
	In 'All Packages' tab, download WordNet, SentiWordNet, Punkt Tokenizer Model, WordNet-InfoContent
	
## 7. Do sentiment analysis
	> python sentiment_analysis.py [input_corpus] [output_directory] [set_size]

	[input_corpus]: review corpus to be analyzed, can be downloaded from https://snap.stanford.edu/data/web-Amazon.html
	[output_directory]: directory that will contain all the *.txt output results, need to create the output folder by yourself before running the script
	[set_size]: the number of reviews that user want to analyze from the entire input_corpus

	Example:

	> python sentiment_analysis.py .\Cell_Phones___Accessories.txt .\result\ 2000
	
## 8. Evaluate performance
	Create a blank gold standard template first
	> python evaluate_sentiment.py init [result] [evalset size] [output_init]
	
	[result]: path to output_sentiment.txt produced by sentiment_analysis.py
	[evalset size]: set size of gold standard
	[output_init]: filename of gold standard output
	
	Example:
	> python evaluate_sentiment.py init .\result\output_sentiment.txt 200 goldstandard.txt
	
	Manually annotate the gold standard
	
	Evaluate the system output to the gold standard
	> python evaluate_sentiment.py eval [goldstandard] [result] [output_eval] [detailed, optional]

	[goldstandard]:	path of manually annotated goldstandard
	[output_eval]: filename of evaluation result output
	
	Example:
	> python evaluate_sentiment.py eval goldstandard.txt .\result\output_sentiment.txt evaluation.txt detailed
