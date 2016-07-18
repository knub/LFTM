package lftm;

import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicAssignment;
import lftm.models.LFDMM;
import lftm.models.LFLDA;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import lftm.utility.CmdArgs;
import lftm.eval.ClusteringEval;

import java.io.File;
import java.util.ArrayList;

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

        ParallelTopicModel tm = ParallelTopicModel.read(
                new File("/home/knub/Repositories/master-thesis/models/topic-models/topic.full.alpha-1-100.256-400.model"));
        System.out.println("Loaded model");

        ArrayList<TopicAssignment> topicAssignments = tm.getData();
        TopicAssignment instance = topicAssignments.get(0);

        System.out.println(instance);
        System.out.println(instance.topicDistribution);
        System.out.println(instance.topicSequence);

        System.exit(1);

        try {

            parser.parseArgument(args);

            switch (cmdArgs.model) {
                case "LFLDA":
                    LFLDA lflda = new LFLDA(cmdArgs.corpus, cmdArgs.vectors, cmdArgs.vocabulary,
                            cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta, cmdArgs.lambda, cmdArgs.initers,
                            cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
                            cmdArgs.initTopicAssgns, cmdArgs.savestep);
                    lflda.inference();
                    break;
                case "LFDMM":
                    LFDMM lfdmm = new LFDMM(cmdArgs.corpus, cmdArgs.vectors, cmdArgs.ntopics,
                            cmdArgs.alpha, cmdArgs.beta, cmdArgs.lambda, cmdArgs.initers,
                            cmdArgs.niters, cmdArgs.twords, cmdArgs.expModelName,
                            cmdArgs.initTopicAssgns, cmdArgs.savestep);
                    lfdmm.inference();
                    break;
                case "Eval":
                    ClusteringEval.evaluate(cmdArgs.labelFile, cmdArgs.dir, cmdArgs.prob);
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

    public static void help(CmdLineParser parser)
    {
        System.out.println("java -jar lftm.LFTM.jar [options ...] [arguments...]");
        parser.printUsage(System.out);
    }
}
