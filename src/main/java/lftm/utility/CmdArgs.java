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

    @Option(name = "-classes", usage = "Specify path to the file containing word vectors", required = true)
    public String classes = "NONE";

    @Option(name = "-ntopics", usage = "Specify number of topics")
    public int ntopics = 256;

    @Option(name = "-alpha", usage = "Specify alpha")
    public double alpha = 0.01;

    @Option(name = "-beta", usage = "Specify beta")
    public double beta = 0.01;

    @Option(name = "-lambda", usage = "Specify mixture weight lambda")
    public double lambda = 0.6;

    @Option(name = "-ndocs", usage = "Specify number of documents")
    public int ndocs = 0;

    @Option(name = "-niters", usage = "Specify number of EM-style sampling iterations")
    public int niters = 200;

    @Option(name = "-twords", usage = "Specify number of top topical words")
    public int twords = 10;

//    @Option(name = "-dir")
//    public String dir = "";
//
//    @Option(name = "-label")
//    public String labelFile = "";
//
//    @Option(name = "-prob")
//    public String prob = "";

    @Option(name = "-sstep")
    public int savestep = 10;

}
