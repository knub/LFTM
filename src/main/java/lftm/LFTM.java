package lftm;

import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicAssignment;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import lftm.models.LFDMM;

import lftm.models.LFLDA;
import lftm.utility.MTRandom;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import lftm.utility.CmdArgs;
import lftm.eval.ClusteringEval;

import java.io.*;
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
public class LFTM
{
    public static void main(String[] args) throws Exception {
        CmdArgs cmdArgs = new CmdArgs();
        CmdLineParser parser = new CmdLineParser(cmdArgs);

        try {
            parser.parseArgument(args);

            switch (cmdArgs.model) {
                case "LFLDA":
                    LFLDA lflda = new LFLDA(cmdArgs.topicModel, cmdArgs.vectors, cmdArgs.vocabulary,
                            cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta, cmdArgs.lambda, cmdArgs.initers,
                            cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName, cmdArgs.savestep);
                    lflda.inference();
                    break;
//                case "LFDMM":
//                    LFDMM lfdmm = new LFDMM(cmdArgs.corpus, cmdArgs.vectors, cmdArgs.ntopics,
//                            cmdArgs.alpha, cmdArgs.beta, cmdArgs.lambda, cmdArgs.initers,
//                            cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
//                            "", 0);
//                    lfdmm.inference();
//                    break;
//                case "Eval":
//                    ClusteringEval.evaluate(cmdArgs.labelFile, cmdArgs.dir, cmdArgs.prob);
//                    break;
                case "preprocess-LFLDA":
                    preprocessLFLDA(cmdArgs.topicModel, cmdArgs.vocabulary);
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

    private static void preprocessLFLDA(String pathToTopicModel, String pathToVectorWords) throws Exception {
        ParallelTopicModel tm = ParallelTopicModel.read(new File(pathToTopicModel));
        HashMap<String, Integer> word2IdVocabulary = new HashMap<>();
        int lastWordId = -1;
        Set<String> vectorWords = getVectorWords(pathToVectorWords);

        StringBuilder sbDocuments = new StringBuilder();

        // for all documents
        Alphabet wordAlphabet = tm.getAlphabet();
        ArrayList<TopicAssignment> data = tm.getData();
        for (TopicAssignment doc : data) {
            int[] wordFeatures = ((FeatureSequence) doc.instance.getData()).getFeatures();
            int[] topicFeatures = doc.topicSequence.getFeatures();
            assert wordFeatures.length == topicFeatures.length :
                    "Document length does not match " + wordFeatures.length + " - " + topicFeatures.length;
            if (wordFeatures.length == 0) {
                System.out.println("Skipping document");
                continue;
            }

            // for all words
            for (int i = 0; i < wordFeatures.length; i += 1) {
                int featureId = wordFeatures[i];
                int topicId = topicFeatures[i];
                String word = (String) wordAlphabet.lookupObject(featureId);
                if (vectorWords.contains(word)) {
                    int wordId;
                    if (word2IdVocabulary.containsKey(word)) {
                        wordId = word2IdVocabulary.get(word);
                    } else {
                        wordId = lastWordId + 1;
                        word2IdVocabulary.put(word, wordId);
                        lastWordId = wordId;
                    }
                    if (i == 0) {
                        sbDocuments.append(wordId).append("-").append(topicId);
                    } else {
                        sbDocuments.append(" ").append(wordId).append("-").append(topicId);
                    }
                }
            }
            sbDocuments.append("\n");
        }
        BufferedWriter bw = new BufferedWriter(new FileWriter(new File(pathToTopicModel + ".lflda")));

        StringBuilder sbAlphabet = new StringBuilder();
        boolean first = true;
        for (Map.Entry<String, Integer> entry : word2IdVocabulary.entrySet()) {
            if (first) {
                sbAlphabet.append(entry.getKey()).append("-").append(entry.getValue());
                first = false;
            } else {
                sbAlphabet.append(" ").append(entry.getKey()).append("-").append(entry.getValue());
            }
        }

        bw.write(sbAlphabet.toString());
        bw.write("\n");
        bw.write(sbDocuments.toString());
        bw.close();
    }

    public static void help(CmdLineParser parser)
    {
        System.out.println("java -jar lftm.LFTM.jar [options ...] [arguments...]");
        parser.printUsage(System.out);
    }
}
