package lftm.models;

import cc.mallet.optimize.InvalidOptimizableException;
import cc.mallet.optimize.Optimizer;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.types.Alphabet;
import cc.mallet.types.AlphabetCarrying;
import cc.mallet.types.MatrixOps;
import com.carrotsearch.hppc.IntArrayList;
import lftm.utility.*;

import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

///**
// * Implementation of the LF-LDA latent feature topic model, using collapsed Gibbs sampling, as
// * described in:
// *
// * Dat Quoc Nguyen, Richard Billingsley, Lan Du and Mark Johnson. 2015. Improving Topic Models with
// * Latent Feature Word Representations. Transactions of the Association for Computational
// * Linguistics, vol. 3, pp. 299-313.
// *
// * @author Dat Quoc Nguyen
// */

@SuppressWarnings("ALL")
public class LFLDA {
    public double alpha; // Hyper-parameter alpha
    public double beta; // Hyper-parameter alpha
    public double alphaSum; // alpha * numTopics
    public double betaSum; // beta * vocabularySize

    public int numTopics; // Number of topics
    public int topWords; // Number of most probable words for each topic

    public double lambda; // Mixture weight value
    public int numIterations; // Number of EM-style sampling iterations

    public List<IntArrayList> corpus; // Word ID-based corpus
    public List<IntArrayList> topicAssignments; // Topics assignments for words
                                                 // in the corpus
    public int numDocuments; // Number of documents in the corpus

    public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
                                                       // given a word
    public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
                                                       // given an ID
    public int vocabularySize; // The number of word types in the corpus

    // numDocuments * numTopics matrix
    // Given a document: number of its words assigned to each topic
    public int[][] docTopicCount;
    // Number of words in every document
    public int[] sumDocTopicCount;
    // numTopics * vocabularySize matrix
    // Given a topic: number of times a word type generated from the topic by
    // the Dirichlet multinomial component
    public int[][] topicWordCountLDA;
    // Total number of words generated from each topic by the Dirichlet
    // multinomial component
    public int[] sumTopicWordCountLDA;
    // numTopics * vocabularySize matrix
    // Given a topic: number of times a word type generated from the topic by
    // the latent feature component
    public int[][] topicWordCountLF;
    // Total number of words generated from each topic by the latent feature
    // component
    public int[] sumTopicWordCountLF;

    // Double array used to sample a topic
    public double[] multiPros;
    // Path to the directory containing the corpus
//    public String folderPath;
    // Path to the topic modeling corpus
    public String topicModelPath;
    public String vectorFilePath;

    public double[][] wordVectors; // Vector representations for words
    public double[][] topicVectors;// Vector representations for topics
    public int vectorSize; // Number of vector dimensions
    public double[][] dotProductValues;
    public double[][] expDotProductValues;
    public double[] sumExpValues; // Partition function values

    public final double l2Regularizer = 0.01; // L2 regularizer value for learning topic vectors
    public final double tolerance = 0.05; // Tolerance value for LBFGS convergence

    public String tAssignsFilePath = "";
    public int savestep = 0;

    private final LFTMTopicModelWriter writer;

    public static boolean alphabetsMatch (AlphabetCarrying object1, AlphabetCarrying object2) {
        Alphabet a1 = object1.getAlphabet();
        Alphabet a2 = object2.getAlphabet();
        if (a1 == null || a2 == null) {
            System.out.println("One is null, the other isn't");
            return false;  // One is null, but the other isn't
        }
        if (!a1.equals(a2)) {
            System.out.println("Does not equal");
//                System.out.println(a1[i]);
//                System.out.println(a2[i]);
            return false;
        }
        return true;
    }

    public LFLDA(String pathToTopicModel, String pathToWordVectorsFile, String pathToVectorWords,
                 int inNumTopics, double inAlpha, double inBeta, double inLambda, int ndocs,
                 int inNumIterations, int inTopWords, int inSaveStep)
        throws Exception {
        alpha = inAlpha;
        beta = inBeta;
        lambda = inLambda;
        numTopics = inNumTopics;
        numIterations = inNumIterations;
        topWords = inTopWords;
        savestep = inSaveStep;
        vectorFilePath = pathToWordVectorsFile;
        topicModelPath = pathToTopicModel;
//        folderPath = topicModelPath.substring(0,
//                Math.max(topicModelPath.lastIndexOf("/"), topicModelPath.lastIndexOf("\\")) + 1);
        numDocuments = ndocs;
        writer = new LFTMTopicModelWriter(this);

        System.out.println("Starting with " + FreeMemory.get(false, 0) + " MB");
        System.out.println("Reading topic model: " + pathToTopicModel);
        TopicModelInfo tm = loadTopicModelInfo(pathToTopicModel, pathToVectorWords);

        corpus = new ArrayList<IntArrayList>(numDocuments);
        topicAssignments = new ArrayList<IntArrayList>(numDocuments);
        vocabularySize = tm.vocabularySize;

        docTopicCount = new int[numDocuments][numTopics];
        sumDocTopicCount = new int[numDocuments];
        topicWordCountLDA = new int[numTopics][vocabularySize];
        sumTopicWordCountLDA = new int[numTopics];
        topicWordCountLF = new int[numTopics][vocabularySize];
        sumTopicWordCountLF = new int[numTopics];

        multiPros = new double[numTopics * 2];
        for (int i = 0; i < numTopics * 2; i++) {
            multiPros[i] = 1.0 / numTopics;
        }

        alphaSum = numTopics * alpha;
        betaSum = vocabularySize * beta;

        System.out.println("Corpus size: " + numDocuments + " docs");
        System.out.println("Vocabulary size: " + vocabularySize);
        System.out.println("Number of topics: " + numTopics);
        System.out.println("alpha: " + alpha);
        System.out.println("beta: " + beta);
        System.out.println("lambda: " + lambda);
        System.out.println("Number of sampling iterations for the LF-LDA model: " + numIterations);
        System.out.println("Number of top topical words: " + topWords);

        readCorpus(pathToTopicModel, tm);

        System.out.println("Reading vectors: " + vectorFilePath);
        readWordVectorsFile(vectorFilePath);

        System.out.println("Initializing embedding datastructures");
        topicVectors = new double[numTopics][vectorSize];
        dotProductValues = new double[numTopics][vocabularySize];
        expDotProductValues = new double[numTopics][vocabularySize];
        sumExpValues = new double[numTopics];
        System.out.println("Memory: " + FreeMemory.get(true, 5) + " MB");
    }

    private void readCorpus(String pathToTopicModel, TopicModelInfo tm) throws IOException {
        int docId = 0;
        Alphabet wordAlphabet = tm.wordAlphabet;
        BufferedReader brAlphabet = new BufferedReader(new FileReader(pathToTopicModel + ".lflda.alphabet"));
        word2IdVocabulary = readWord2IdVocabulary(brAlphabet.readLine());
        id2WordVocabulary = buildId2WordVocabulary(word2IdVocabulary);
        BufferedReader brDocument = new BufferedReader(new FileReader(pathToTopicModel + ".lflda"));
        int lineNr = 0;
        IntArrayList document = new IntArrayList();
        IntArrayList topics = new IntArrayList();

        // for all documents
        for (String line; (line = brDocument.readLine()) != null;) {
            if (line.equals("##")) {
                if (document.size() > 0) {
                    corpus.add(document);
                    topicAssignments.add(topics);
                    document = new IntArrayList();
                    topics = new IntArrayList();
                    docId += 1;
                    if (docId % 100000 == 0) {
                        System.out.println(docId);
                    }
                }
                continue;
            }
            try {
                int wordId = Integer.parseInt(line.substring(0, 6));
                int topicId = Integer.parseInt(line.substring(7, 13));

                // Topic initialization
                int topicOffset = MTRandom.nextDouble() < lambda ? 0 : numTopics;
                int subtopic = topicId + topicOffset;
                int topic = subtopic % numTopics;
                if (topic == subtopic) { // Generated from the latent feature component
                    topicWordCountLF[topic][wordId] += 1;
                    sumTopicWordCountLF[topic] += 1;
                } else {// Generated from the Dirichlet multinomial component
                    topicWordCountLDA[topic][wordId] += 1;
                    sumTopicWordCountLDA[topic] += 1;
                }
                docTopicCount[docId][topic] += 1;
                sumDocTopicCount[docId] += 1;

                document.add(wordId);
                topics.add(subtopic);
            } catch (Exception e) {
                System.out.println(line);
                System.out.println("lineNr = " + lineNr);
                System.out.println("line = <" + line + ">");
                throw e;
            }
            lineNr += 1;
        }
    }

    private HashMap<Integer, String> buildId2WordVocabulary(HashMap<String, Integer> word2IdVocabulary) {
        HashMap<Integer, String> result = new HashMap<>();
        for (Map.Entry<String, Integer> entry : word2IdVocabulary.entrySet()) {
            result.put(entry.getValue(), entry.getKey());
        }
        return result;
    }

    private HashMap<String,Integer> readWord2IdVocabulary(String line) {
        HashMap<String, Integer> result = new HashMap<>();
        String[] split = line.split(" ");
        for (String alphabetEntry : split) {
            String[] innerSplit = alphabetEntry.split("#");
            result.put(innerSplit[0], Integer.parseInt(innerSplit[1]));

        }
        return result;
    }

    class TopicModelInfo {
        public TopicModelInfo(Alphabet wordAlphabet, int vocabularySize) {
            this.wordAlphabet = wordAlphabet;
            this.vocabularySize = vocabularySize;
        }

        public Alphabet wordAlphabet;
        public int vocabularySize;
    }
    private TopicModelInfo loadTopicModelInfo(String pathToTopicModel, String pathToVectorWords) throws Exception {
        System.out.println("Loading topic model info " + pathToTopicModel);
        ParallelTopicModel tm = ParallelTopicModel.read(
                new File(pathToTopicModel));
        System.out.println("Memory: " + FreeMemory.get(true, 15) + " MB");

        Alphabet alp = tm.getAlphabet();
        vocabularySize = determineVocabularySize(tm, getVectorWords(pathToVectorWords));
        return new TopicModelInfo(alp, vocabularySize);
    }
    private static Set<String> getVectorWords(String pathToVectorWords) throws Exception {
        Set<String> vectorWords = new HashSet<>();
        BufferedReader br = new BufferedReader(new FileReader(pathToVectorWords));
        for (String word; (word = br.readLine()) != null;) {
            vectorWords.add(word);
        }
        return vectorWords;
    }

    private int determineVocabularySize(ParallelTopicModel tm, Set<String> vectorWords) throws Exception {
        int size = 0;
        Iterator it = tm.getAlphabet().iterator();
        while (it.hasNext()) {
            String w = (String) it.next();
            if (vectorWords.contains(w)) {
                size += 1;
            }
        }
        it = tm.getAlphabet().iterator();
        return size;
    }

    public void readWordVectorsFile(String pathToWordVectorsFile)
        throws Exception
    {
        System.out.println("Reading word vectors from word-vectors file " + pathToWordVectorsFile
                + "...");

        BufferedReader br = new BufferedReader(new FileReader(pathToWordVectorsFile));
        String[] elements = br.readLine().trim().split("\\s+");
        vectorSize = elements.length - 1;
        System.out.println("Word vector size: " + vectorSize);
        System.out.println("Vocabulary size: " + vocabularySize);
        wordVectors = new double[vocabularySize][vectorSize];
        String word = elements[0];
        if (word2IdVocabulary.containsKey(word)) {
            for (int j = 0; j < vectorSize; j++) {
                wordVectors[word2IdVocabulary.get(word)][j] = new Double(elements[j + 1]);
            }
        }
        for (String line; (line = br.readLine()) != null;) {
            elements = line.trim().split("\\s+");
            word = elements[0];
            if (word2IdVocabulary.containsKey(word)) {
                for (int j = 0; j < vectorSize; j++) {
                    wordVectors[word2IdVocabulary.get(word)][j] = new Double(elements[j + 1]);
                }
            }
        }

        for (int i = 0; i < vocabularySize; i++) {
            if (MatrixOps.absNorm(wordVectors[i]) == 0.0) {
                String w = id2WordVocabulary.get(i);
                w = w == null ? "<null>" : w;
                throw new Exception("The word \"" + w + "\" at index " + i + " doesn't have a corresponding vector!!!");
            }
        }
    }

    public void inference() throws IOException
    {
        System.out.println("Running Gibbs sampling inference: ");

        for (int iter = 1; iter <= numIterations; iter++) {

            System.out.println("\tLFLDA sampling iteration: " + iter);
            optimizeTopicVectors();
            sampleSingleIteration();

            if (savestep > 0 && iter % savestep == 0 && iter < numIterations) {
                System.out.println("\t\tSaving the output from the " + iter + "^{th} sample");
                writer.write(String.valueOf(iter));
            }
        }

        writer.writeParameters();
        System.out.println("Writing output from the last sample ...");
        writer.write("final");

        System.out.println("Sampling completed!");
    }

    public void optimizeTopicVectors()
    {
        System.out.println("\t\tEstimating topic vectors ...");
        System.out.println(new Date());
        sumExpValues = new double[numTopics];
        dotProductValues = new double[numTopics][vocabularySize];
        expDotProductValues = new double[numTopics][vocabularySize];

        final AtomicInteger finishedTopics = new AtomicInteger(0);
        System.out.println("Memory: " + FreeMemory.get(true, 5) + " MB");
        Parallel.loop(numTopics, new Parallel.LoopInt()
        {
            @Override
            public void compute(int topic)
            {
                int rate = 1;
                boolean check = true;
                while (check) {
                    double l2Value = l2Regularizer * rate;
                    try {
                        TopicVectorOptimizer optimizer = new TopicVectorOptimizer(
                                topicVectors[topic], topicWordCountLF[topic], wordVectors, l2Value);

                        Optimizer gd = new LBFGS(optimizer, tolerance);
                        gd.optimize(500);
                        optimizer.getParameters(topicVectors[topic]);
                        sumExpValues[topic] = optimizer.computePartitionFunction(
                                dotProductValues[topic], expDotProductValues[topic]);
                        check = false;

                        if (sumExpValues[topic] == 0 || Double.isInfinite(sumExpValues[topic])) {
                            double max = -1000000000.0;
                            for (int index = 0; index < vocabularySize; index++) {
                                if (dotProductValues[topic][index] > max)
                                    max = dotProductValues[topic][index];
                            }
                            for (int index = 0; index < vocabularySize; index++) {
                                expDotProductValues[topic][index] = Math
                                        .exp(dotProductValues[topic][index] - max);
                                sumExpValues[topic] += expDotProductValues[topic][index];
                            }
                        }
                    }
                    catch (InvalidOptimizableException e) {
                        System.out.print("ERROR-" + topic);
                        e.printStackTrace();
                        check = true;
                    }
                    rate = rate * 10;
                }
                int currentValue = finishedTopics.incrementAndGet();
                double percentage = ((double) currentValue) / numTopics;
                System.out.print(String.format("%.2f ", percentage));
                System.out.flush();
            }
        });
        System.out.println();
        System.out.println(new Date());
    }

    public void sampleSingleIteration()
    {
        System.out.println("\t\tRunning iteration ...");
        System.out.println(new Date());
        for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
            if (dIndex % 100000 == 0) {
                System.out.print(dIndex + " ");
                System.out.flush();
            }
            int docSize = corpus.get(dIndex).size();
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                int word = corpus.get(dIndex).get(wIndex);// wordID
                int subtopic = topicAssignments
                        .get(dIndex)
                        .get(wIndex);
                int topic = subtopic % numTopics;

                docTopicCount[dIndex][topic] -= 1;
                if (topic == subtopic) { // LF(w|t) + LDA(t|d)
                    topicWordCountLF[topic][word] -= 1;
                    sumTopicWordCountLF[topic] -= 1;
                }
                else { // LDA(w|t) + LDA(t|d)
                    topicWordCountLDA[topic][word] -= 1;
                    sumTopicWordCountLDA[topic] -= 1;
                }

                // Sample a pair of topic z and binary indicator variable s
                for (int tIndex = 0; tIndex < numTopics; tIndex++) {

                    multiPros[tIndex] = (docTopicCount[dIndex][tIndex] + alpha) * lambda
                            * expDotProductValues[tIndex][word] / sumExpValues[tIndex];

                    multiPros[tIndex + numTopics] = (docTopicCount[dIndex][tIndex] + alpha)
                            * (1 - lambda) * (topicWordCountLDA[tIndex][word] + beta)
                            / (sumTopicWordCountLDA[tIndex] + betaSum);

                }
                subtopic = FuncUtils.nextDiscrete(multiPros);
                topic = subtopic % numTopics;

                docTopicCount[dIndex][topic] += 1;
                if (topic == subtopic) {
                    topicWordCountLF[topic][word] += 1;
                    sumTopicWordCountLF[topic] += 1;
                }
                else {
                    topicWordCountLDA[topic][word] += 1;
                    sumTopicWordCountLDA[topic] += 1;
                }
                // Update topic assignments
                topicAssignments.get(dIndex).set(wIndex, subtopic);
            }
        }
        System.out.println(new Date());
    }

}
