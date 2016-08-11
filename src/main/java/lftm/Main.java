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
                    preprocessLFLDA(cmdArgs.topicModel, cmdArgs.vectors, cmdArgs.classes);
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

    private static void preprocessLFLDA(String pathToTopicModel, String pathToEmbeddings, String classesFile) throws Exception {
        ParallelTopicModel tm = ParallelTopicModel.read(new File(pathToTopicModel));
        HashMap<String, Integer> word2IdVocabulary = new HashMap<>();
        int lastWordId = -1;
        Set<String> vectorWords = getVectorWords(pathToEmbeddings + ".vocab");

        String embeddingFileName = Paths.get(pathToEmbeddings).getFileName().toString();
        PrintWriter pwDocuments = new PrintWriter(new BufferedWriter(new FileWriter(new File(
                pathToTopicModel + "." + embeddingFileName + ".restricted"))));
        PrintWriter pwAlphabet = new PrintWriter(new BufferedWriter(new FileWriter(new File(
                pathToTopicModel + "." + embeddingFileName + ".restricted.alphabet"))));
        PrintWriter pwClasses = new PrintWriter(new BufferedWriter(new FileWriter(new File(
                pathToTopicModel + "." + embeddingFileName + ".restricted.classes"))));

        if (classesFile.equals("NONE"))
            throw new Exception("Must set classes file!");
        BufferedReader classReader = new BufferedReader(new FileReader(classesFile));

        // for all documents
        Alphabet wordAlphabet = tm.getAlphabet();
        ArrayList<TopicAssignment> data = tm.getData();
        for (TopicAssignment doc : data) {
            String clazz = classReader.readLine();

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
                pwClasses.println(clazz);
                pwDocuments.println("##");
            }
        }

        final boolean[] first = { true };
        word2IdVocabulary.entrySet().stream().sorted(Map.Entry.comparingByValue()).forEach(entry -> {
            if (first[0]) {
                pwAlphabet.write(entry.getKey() + "#" + entry.getValue());
                first[0] = false;
            } else {
                pwAlphabet.write("\n" + entry.getKey() + "#" + entry.getValue());
            }

        });
        pwDocuments.close();
        pwAlphabet.close();
        pwClasses.close();
    }

    public static void help(CmdLineParser parser)
    {
        System.out.println("java -jar lftm.Main.jar [options ...] [arguments...]");
        parser.printUsage(System.out);
    }
}
