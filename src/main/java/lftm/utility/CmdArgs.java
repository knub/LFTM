package lftm.utility;

import org.kohsuke.args4j.Option;

public class CmdArgs
{

    @Option(name = "-model", usage = "Specify model", required = true)
    public String model = "";

    @Option(name = "-topicmodel", usage = "Specify path to topic model", required = true)
    public String topicModel = "";

    @Option(name = "-vectors", usage = "Specify path to the file containing word vectors", required = true)
    public String vectors = "";

    @Option(name = "-vocabulary", usage = "Specify vocabulary to use", required = true)
    public String vocabulary = "";

    @Option(name = "-ntopics", usage = "Specify number of topics", required = true)
    public int ntopics = 20;

    @Option(name = "-alpha", usage = "Specify alpha", required = true)
    public double alpha = 0.1;

    @Option(name = "-beta", usage = "Specify beta", required = true)
    public double beta = 0.01;

    @Option(name = "-lambda", usage = "Specify mixture weight lambda", required = true)
    public double lambda = 0.6;

    @Option(name = "-initers", usage = "Specify number of initial sampling iterations", required = true)
    public int initers = 2000;

    @Option(name = "-niters", usage = "Specify number of EM-style sampling iterations", required = true)
    public int niters = 200;

    @Option(name = "-twords", usage = "Specify number of top topical words", required = true)
    public int twords = 20;

    @Option(name = "-name", usage = "Specify a name to a topic modeling experiment", required = true)
    public String expModelName = "model";

//    @Option(name = "-dir")
//    public String dir = "";
//
//    @Option(name = "-label")
//    public String labelFile = "";
//
//    @Option(name = "-prob")
//    public String prob = "";

    @Option(name = "-sstep", required = true)
    public int savestep = 0;

}
