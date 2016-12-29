import java.util.*;
import java.io.*;
import java.nio.file.Files;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.OriginalTextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.util.CoreMap;

import edu.stanford.nlp.parser.lexparser.*;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;

/**
 * Author: Xinyu Hua
 * Date:   12/29/2016
 * Functions:   
 * 1    SentenceSplit
 * 2    Tokenizer
 * 3    Lemmatizer
 * 4    Postag
 * 5    Parse
 *
 */

public class StanfordNLPRunner{
	 	
		protected StanfordCoreNLP pipeline;
		MaxentTagger tagger =  new MaxentTagger("lib/english-left3words-distsim.tagger");
        
        public static String[] special_symbols = {"'s","'ve","'m","'d","n't","'re"};
        static LexicalizedParser lp;
        static TreebankLanguagePack tlp;
        static GrammaticalStructureFactory gsf;

	    public StanfordNLPRunner() {
	        // Create StanfordCoreNLP object properties, with POS tagging
	        // (required for lemmatization), and lemmatization
	        Properties props;
	        props = new Properties();
	        props.put("annotators", "tokenize, ssplit, pos, lemma");
	
	        // StanfordCoreNLP loads a lot of models, so you probably
	        // only want to do this once per execution
	        this.pipeline = new StanfordCoreNLP(props);
	    }
	
        public static void main(String[] args)throws Exception{
            testParser();
        }


        // 1 Sentence split

        public static void testSplitSentence()throws Exception{
            splitSentenceForFile("a.txt","b.txt");
        }

        public static void splitSentenceForFile(String PATH_IN, String PATH_OUT)throws Exception{
            BufferedReader fileReader = new BufferedReader(new FileReader( PATH_IN ));
            BufferedWriter fileWriter = new BufferedWriter(new FileWriter( PATH_OUT));
            String line;
            String paragraph = "";
            while((line = fileReader.readLine())!=null){
                paragraph += " " + line;
            }
            Reader reader = new StringReader(paragraph);
            DocumentPreprocessor dp = new DocumentPreprocessor( reader );
            List<String> sentenceList = new ArrayList<>();

            for( List<HasWord> sentence : dp ) {
                String sentenceString = Sentence.listToString(sentence);
                sentenceList.add(sentenceString);
            }

            for( String sentence : sentenceList) {
                fileWriter.append( sentence.trim() + "\n");
            }
            fileWriter.close();
            fileReader.close();
        }

        // 2 Tokenizer
        public static void testTokenizer()throws Exception{
            tokenizeForFile("a.txt","b.txt");
        }

        public static void tokenizeForFile(String PATH_IN, String PATH_OUT) throws Exception{
            BufferedWriter fileWriter = new BufferedWriter(new FileWriter( PATH_OUT));
            PTBTokenizer<CoreLabel> ptbt = new PTBTokenizer<>(new FileReader(PATH_IN), new CoreLabelTokenFactory(), "");

            while(ptbt.hasNext()){
                CoreLabel label = ptbt.next();
                fileWriter.append( label + "\n");
            }
            fileWriter.close();
        }


        // 3 Lemmatizer( also produce tokenization )
        public static void testLemmatizer()throws Exception{
            lemmatizeFile("a.txt", "b.txt");
        }

        public static void lemmatizeFile(String PATH_IN, String PATH_OUT) throws Exception{
            StanfordNLPRunner snr = new StanfordNLPRunner();
            BufferedReader fileReader = new BufferedReader(new FileReader(PATH_IN));
            BufferedWriter fileWriter = new BufferedWriter(new FileWriter(PATH_OUT));
            String line;
            while((line = fileReader.readLine())!=null){
                List<List<String>> rst = snr.lemmatize(line);
                List<String> lemmas = rst.get(0);
                for(String lemma : lemmas){
                    fileWriter.append(lemma + " ");
                }
                fileWriter.append("\n");
            }
            fileReader.close();
            fileWriter.close();
        }

        public List<List<String>> lemmatize(String doc)throws Exception{
            List<String> lemmas = new LinkedList<>();
            List<String> original = new LinkedList<>();
            List<List<String>> result = new ArrayList<List<String>>();
            Annotation document = new Annotation(doc);
            this.pipeline.annotate(document);

            List<CoreMap> sentences = document.get(SentencesAnnotation.class);
            for(CoreMap sentence : sentences) {
                for(CoreLabel token: sentence.get(TokensAnnotation.class)){
                    original.add(token.get(OriginalTextAnnotation.class));
                    lemmas.add(token.get(LemmaAnnotation.class));
                }
            }
            result.add(lemmas);
            result.add(original);
            return result;
        }


        // 3 Postag
        public static void testPosTag()throws Exception{
            posTagForFile("a.txt","b.txt");
        }

        public static void posTagForFile(String PATH_IN, String PATH_OUT)throws Exception{
            StanfordNLPRunner snr = new StanfordNLPRunner();
            BufferedReader fileReader = new BufferedReader(new FileReader(PATH_IN));
            BufferedWriter fileWriter = new BufferedWriter(new FileWriter(PATH_OUT));
            String line;
            while((line = fileReader.readLine())!=null){
                List<String> rst = snr.posTag(line);
                for(String tag : rst){
                    fileWriter.append(tag + " ");
                }
                fileWriter.append("\n");
            }
            fileReader.close();
            fileWriter.close();
        }

	    public List<String> posTag(String sentence)
	    {
	    	String tagged = tagger.tagString(sentence);
	    	List<String> tags = Arrays.asList(tagged.split("\\s+"));
	        return tags;
	    }

        // 4 Parsing
        public static void testParser()throws Exception{
            parser();

            String[] rst3 = dependencyParse("he is not here to eat the food.");


            for(String s : rst3){
                System.out.print(s + " ");
            }
     
        }

        public static void parseForFile(String PATH_IN, String PATH_OUT)throws Exception{
            parser();
            BufferedReader fileReader = new BufferedReader(new FileReader(PATH_IN));
            BufferedWriter fileWriter = new BufferedWriter(new FileWriter(PATH_OUT));
            String line;
            while((line = fileReader.readLine())!=null){
                String[] rst = dependencyparse(line);
                for(String tag : rst){
                    fileWriter.append(tag + " ");
                }
                fileWriter.append("\n");
            }
            fileReader.close();
            fileWriter.close();
        }

        public static void parser()throws Exception{
            lp = LexicalizedParser.loadModel(
                 "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz",
                 "-maxLength", "80", "-retainTmpSubcategories");
            tlp = new PennTreebankLanguagePack();
            gsf = tlp.grammaticalStructureFactory();

        }

		public static String[] dependencyParse(String sentence){
		String[] result;
		String[] sent = sentence.split("((\\s+)|(?<=,)|(?=,)|(?=\\.))");
		Tree parse = lp.apply(Sentence.toWordList(sent));
		GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
		Collection<TypedDependency> tdl = gs.typedDependencies();
		result = new String[tdl.size()];
		int i = 0;
		for(TypedDependency t : tdl){
			result[ i++ ] = t.toString();
		}
		return result;
	}
	
	
}
