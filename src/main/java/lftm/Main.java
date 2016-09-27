package lftm;

import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicAssignment;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;

import lftm.models.LFLDA;
import org.apache.commons.lang3.text.WordUtils;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import lftm.utility.CmdArgs;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;

/**
 * Implementations of the LF-LDA and LF-DMM latent feature topic models, using collapsed Gibbs
 * sampling, as described in:
 * 
 * Dat Quoc Nguyen, Richard Billingsley, Lan Du and Mark Johnson. 2015. Improving Topic Models with
 * Latent Feature Word Representations. Transactions of the Association for Computational
 * Linguistics, vol. 3, pp. 299-313.
 * 
 * @author Dat Quoc Nguyen
 * 
 */


@SuppressWarnings("WeakerAccess")
public class Main
{
    public static void main(String[] args) throws Exception {
        CmdArgs cmdArgs = new CmdArgs();
        CmdLineParser parser = new CmdLineParser(cmdArgs);

        try {
            parser.parseArgument(args);

            switch (cmdArgs.model) {
                case "LFLDA":

                    LFLDA lflda = new LFLDA(cmdArgs.topicModel, cmdArgs.vectors,
                            cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta, cmdArgs.lambda, cmdArgs.ndocs,
                            cmdArgs.niters, cmdArgs.twords, cmdArgs.savestep);
                    lflda.inference();
                    break;
//                case "Eval":
//                    ClusteringEval.evaluate(cmdArgs.labelFile, cmdArgs.dir, cmdArgs.prob);
//                    break;
                case "preprocess-LFLDA":
//                    if (cmdArgs.topicModel.contains("sentences")) {
//                        System.out.println("Using sentences parsing");
//                        preprocessLFLDASentences(cmdArgs.topicModel, cmdArgs.vectors, cmdArgs.classes);
//                    } else {
                        System.out.println("Using articles parsing");
                        preprocessLFLDA(cmdArgs.topicModel, cmdArgs.vectors);
//                    }
                    break;
                default:
                    System.out
                            .println("Error: Option \"-model\" must get \"LFLDA\" or \"LFDMM\" or \"Eval\"");
                    System.out.println("\tLFLDA: Specify the LF-LDA topic model");
                    System.out.println("\tLFDMM: Specify the LF-DMM topic model");
                    System.out.println("\tEval: Specify the document clustering evaluation");
                    help(parser);
            }
        }
        catch (CmdLineException cle) {
            System.out.println("Error: " + cle.getMessage());
            help(parser);
        }
        catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static Set<String> getVectorWords(String pathToVectorWords) throws Exception {
        Set<String> vectorWords = new HashSet<>();
        BufferedReader br = new BufferedReader(new FileReader(pathToVectorWords));
        for (String word; (word = br.readLine()) != null;) {
            vectorWords.add(word);
        }
        return vectorWords;
    }

    private static void preprocessLFLDA(String pathToTopicModel, String pathToEmbeddings) throws Exception {
        ParallelTopicModel tm = ParallelTopicModel.read(new File(pathToTopicModel));
        HashMap<String, Integer> word2IdVocabulary = new HashMap<>();
        int lastWordId = -1;
        Set<String> vectorWords = getVectorWords(pathToEmbeddings + ".vocab");
        System.out.println("Vocabulary contains " + vectorWords.size() + " words!");

        String embeddingFileName = Paths.get(pathToEmbeddings).getFileName().toString();
        PrintWriter pwDocuments = new PrintWriter(new BufferedWriter(new FileWriter(new File(
                pathToTopicModel + "." + embeddingFileName + ".restricted"))));
        PrintWriter pwAlphabet = new PrintWriter(new BufferedWriter(new FileWriter(new File(
                pathToTopicModel + "." + embeddingFileName + ".restricted.vocab"))));
        PrintWriter pwAlphabetWithCounts = new PrintWriter(new BufferedWriter(new FileWriter(new File(
                pathToTopicModel + "." + embeddingFileName + ".restricted.vocab.counts"))));
        PrintWriter pwClasses = null;

        boolean doTrackClasses = tm.getData().get(0).instance.getTarget() != null;
        if (doTrackClasses) {
            System.out.println("Tracking classes");
            pwClasses = new PrintWriter(new BufferedWriter(new FileWriter(new File(
                    pathToTopicModel + "." + embeddingFileName + ".restricted.classes"))));
        } else {
            System.out.println("NOT TRACKING CLASSES");
        }


        // for all documents
        Alphabet wordAlphabet = tm.getAlphabet();
        ArrayList<TopicAssignment> data = tm.getData();
        System.out.println("There are " + data.size() + " documents before");
        Counter<String> c = new Counter<>();
        for (TopicAssignment doc : data) {
            String clazz = doc.instance.getTarget().toString();

            int[] wordFeatures = ((FeatureSequence) doc.instance.getData()).getFeatures();
            int[] topicFeatures = doc.topicSequence.getFeatures();
            assert wordFeatures.length == topicFeatures.length :
                    "Document length does not match " + wordFeatures.length + " - " + topicFeatures.length;
            if (wordFeatures.length == 0) {
                System.out.println("Skipping document");
                continue;
            }

            boolean atLeastOneWord = false;
            // for all words
            for (int i = 0; i < wordFeatures.length; i += 1) {
                int featureId = wordFeatures[i];
                int topicId = topicFeatures[i];
                String word = (String) wordAlphabet.lookupObject(featureId);
                if (vectorWords.contains(word) ||
                        vectorWords.contains(word.toUpperCase()) ||
                        vectorWords.contains(WordUtils.capitalize(word))) {
                    atLeastOneWord = true;
                    c.add(word);
                    int wordId;
                    if (word2IdVocabulary.containsKey(word)) {
                        wordId = word2IdVocabulary.get(word);
                    } else {
                        wordId = lastWordId + 1;
                        word2IdVocabulary.put(word, wordId);
                        lastWordId = wordId;
                    }
                    pwDocuments.println(String.format("%06d#%06d", wordId, topicId));
//                    pwDocuments.print(word + " ");
                } else {
//                    System.out.println("Ignoring word: " + word);
                }
            }
            if (atLeastOneWord) {
                if (doTrackClasses)
                    pwClasses.println(clazz);
                pwDocuments.println("##");
            }
        }

        final boolean[] first = { true };
        word2IdVocabulary.entrySet().stream().sorted(Map.Entry.comparingByValue()).forEach(entry -> {
            String word = entry.getKey();
            int wordCount = c.count(word);
            double wordLogProb = Math.log((double) wordCount / c.totalCount());
            if (first[0]) {
                pwAlphabet.write(word);
                pwAlphabetWithCounts.write(String.format("%s\t%d\t%f", word, wordCount, wordLogProb));
                first[0] = false;
            } else {
                pwAlphabet.write("\n" + entry.getKey());
                pwAlphabetWithCounts.write(String.format("%n%s\t%d\t%f", word, wordCount, wordLogProb));
            }

        });
        pwDocuments.close();
        pwAlphabet.close();
        pwAlphabetWithCounts.close();
        if (doTrackClasses)
            pwClasses.close();
    }


    private static void preprocessLFLDASentences(String pathToTopicModel, String pathToEmbeddings, String classesFile) throws Exception {
        ParallelTopicModel tm = ParallelTopicModel.read(new File(pathToTopicModel));
        HashMap<String, Integer> word2IdVocabulary = new HashMap<>();
        int lastWordId = -1;
        Set<String> vectorWords = getVectorWords(pathToEmbeddings + ".vocab");

        String embeddingFileName = Paths.get(pathToEmbeddings).getFileName().toString();
        PrintWriter pwDocuments = new PrintWriter(new BufferedWriter(new FileWriter(new File(
                pathToTopicModel + "." + embeddingFileName + ".restricted"))));
        PrintWriter pwAlphabet = new PrintWriter(new BufferedWriter(new FileWriter(new File(
                pathToTopicModel + "." + embeddingFileName + ".restricted.vocab"))));
        PrintWriter pwAlphabetWithCounts = new PrintWriter(new BufferedWriter(new FileWriter(new File(
                pathToTopicModel + "." + embeddingFileName + ".restricted.vocab.counts"))));
        PrintWriter pwClasses = new PrintWriter(new BufferedWriter(new FileWriter(new File(
                pathToTopicModel + "." + embeddingFileName + ".restricted.classes"))));

        if (!classesFile.equals("NONE"))
            throw new Exception("Must be set to NONE");

        // for all documents
        Alphabet wordAlphabet = tm.getAlphabet();
        ArrayList<TopicAssignment> data = tm.getData();
        System.out.println("There are " + data.size() + " documents");
        Counter<String> c = new Counter<>();

        Integer lastName = (Integer) data.get(0).instance.getName();
        Integer lastTarget = (Integer) data.get(0).instance.getTarget();

        int k = 0;
        boolean atLeastOneWordInDoc = false;
        for (TopicAssignment sentence : data) {
            Integer target = (Integer) sentence.instance.getTarget();
            Integer name = (Integer) sentence.instance.getName();
            if (name == null || target == null) {
                throw new RuntimeException("Name or target is null " + String.valueOf(name) + String.valueOf(target));
            }

            int[] wordFeatures = ((FeatureSequence) sentence.instance.getData()).getFeatures();
            int[] topicFeatures = sentence.topicSequence.getFeatures();
            assert wordFeatures.length == topicFeatures.length :
                    "Document length does not match " + wordFeatures.length + " - " + topicFeatures.length;
            if (wordFeatures.length == 0) {
                System.out.println("Skipping sentence");
                continue;
            }

            boolean atLeastOneWord = false;
            // for all words
            for (int i = 0; i < wordFeatures.length; i += 1) {
                int featureId = wordFeatures[i];
                int topicId = topicFeatures[i];
                String word = (String) wordAlphabet.lookupObject(featureId);
                if (vectorWords.contains(word) ||
                        vectorWords.contains(word.toUpperCase()) ||
                        vectorWords.contains(WordUtils.capitalize(word))) {
                    if (!atLeastOneWord) {
                        pwDocuments.println(name.toString() + "\t" + target.toString());
                    }
                    atLeastOneWord = true;
                    c.add(word);
                    int wordId;
                    if (word2IdVocabulary.containsKey(word)) {
                        wordId = word2IdVocabulary.get(word);
                    } else {
                        wordId = lastWordId + 1;
                        word2IdVocabulary.put(word, wordId);
                        lastWordId = wordId;
                    }
                    pwDocuments.println(String.format("%06d#%06d", wordId, topicId));
                }
            }
            if (atLeastOneWord) {
                pwDocuments.println("##");
            }
            atLeastOneWordInDoc = atLeastOneWordInDoc || atLeastOneWord;
            if (!name.equals(lastName)) {
//                k += 1;
//                System.out.println(String.format("%d %d %d", k, name, lastName));
                lastName = name;
                if (atLeastOneWordInDoc) {
                    pwClasses.println(lastTarget);
                    lastTarget = target;
                    atLeastOneWordInDoc = false;
                }
            }
        }
        if (atLeastOneWordInDoc) {
            pwClasses.println(lastTarget);
        }

        final boolean[] first = { true };
        word2IdVocabulary.entrySet().stream().sorted(Map.Entry.comparingByValue()).forEach(entry -> {
            String word = entry.getKey();
            int wordCount = c.count(word);
            double wordLogProb = Math.log((double) wordCount / c.totalCount());
            if (first[0]) {
                pwAlphabet.write(word);
                pwAlphabetWithCounts.write(String.format("%s\t%d\t%f", word, wordCount, wordLogProb));
                first[0] = false;
            } else {
                pwAlphabet.write("\n" + entry.getKey());
                pwAlphabetWithCounts.write(String.format("%n%s\t%d\t%f", word, wordCount, wordLogProb));
            }

        });
        pwDocuments.close();
        pwAlphabet.close();
        pwAlphabetWithCounts.close();
        pwClasses.close();
    }

    public static void help(CmdLineParser parser)
    {
        System.out.println("java -jar lftm.Main.jar [options ...] [arguments...]");
        parser.printUsage(System.out);
    }
}

class Counter<T> {
    final Map<T, Integer> counts = new HashMap<>();
    private int totalCount = 0;

    public void add(T t) {
        counts.merge(t, 1, Integer::sum);
        totalCount += 1;
    }

    public int totalCount() {
        return totalCount;
    }

    public int count(T t) {
        return counts.getOrDefault(t, 0);
    }
}
