package lftm.utility;

import lftm.models.LFLDA;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public class LFLDATopicModelWriter {

    private final LFLDA lflda;
    private final String baseName;

    public LFLDATopicModelWriter(LFLDA lflda ) {
        this.lflda = lflda;
        String topicFolder = new File(lflda.topicModelPath).getParent();
        File resultsFile = new File(String.format("%s/lflda.dim-%d.lambda-%s.alphasum-%s.beta-%s",
                topicFolder,
                lflda.numDimensions,
                String.valueOf(lflda.lambda).replace('.', '-'),
                String.valueOf(lflda.alphaSum).replace('.', '-'),
                String.valueOf(lflda.beta).replace('.', '-')
        ));
        resultsFile.mkdir();
        baseName = resultsFile.getAbsolutePath() + "/";
        System.out.println("Output in " + baseName);
    }
    public void writeParameters() throws IOException
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(baseName + "params"));
        writer.write("-model" + "\t" + "LFLDA");
        writer.write("\n-topicmodel" + "\t" + lflda.topicModelPath);
        writer.write("\n-vectors" + "\t" + lflda.vectorFilePath);
        writer.write("\n-ntopics" + "\t" + lflda.numTopics);
        writer.write("\n-alpha-sum" + "\t" + lflda.alphaSum);
        writer.write("\n-beta" + "\t" + lflda.beta);
        writer.write("\n-lambda" + "\t" + lflda.lambda);
        writer.write("\n-niters" + "\t" + lflda.numIterations);
        writer.write("\n-twords" + "\t" + lflda.topWords);
        if (lflda.savestep > 0)
            writer.write("\n-sstep" + "\t" + lflda.savestep);

        writer.close();
    }

    public void writeTopicAssignments(String name) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(baseName + name + ".topic-assignments"));
        for (int dIndex = 0; dIndex < lflda.numDocuments; dIndex++) {
            int docSize = lflda.corpus.get(dIndex).size();
            for (int wIndex = 0; wIndex < docSize; wIndex++) {
                writer.write(lflda.topicAssignments.get(dIndex).get(wIndex) + " ");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void writeTopicVectors(String name) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(baseName + name + ".topic-vectors"));
        for (int i = 0; i < lflda.numTopics; i++) {
            for (int j = 0; j < lflda.vectorSize; j++)
                writer.write(lflda.topicVectors[i][j] + " ");
            writer.write("\n");
        }
        writer.close();
    }

    public void writeTopTopicalWords(String name) throws IOException {
        BufferedWriter writer = lflda.topWords == 10 ?
                new BufferedWriter(new FileWriter(baseName + name + ".topics")) :
                new BufferedWriter(new FileWriter(baseName + name + "." + lflda.topWords + ".topics"));

        for (int tIndex = 0; tIndex < lflda.numTopics; tIndex++) {
            writer.write(String.valueOf(tIndex));

            Map<Integer, Double> topicWordProbs = new TreeMap<Integer, Double>();
            for (int wIndex = 0; wIndex < lflda.vocabularySize; wIndex++) {

                double pro = lflda.lambda * lflda.expDotProductValues[tIndex][wIndex] / lflda.sumExpValues[tIndex]
                        + (1 - lflda.lambda) * (lflda.topicWordCountLDA[tIndex][wIndex] + lflda.beta)
                        / (lflda.sumTopicWordCountLDA[tIndex] + lflda.betaSum);

                topicWordProbs.put(wIndex, pro);
            }
            topicWordProbs = FuncUtils.sortByValueDescending(topicWordProbs);

            Set<Integer> mostLikelyWords = topicWordProbs.keySet();
            int count = 0;
            for (Integer index : mostLikelyWords) {
                if (count < lflda.topWords) {
                    writer.write(" " + lflda.id2WordVocabulary.get(index));
                    count += 1;
                }
                else {
                    writer.write("\n");
                    break;
                }
            }
        }
        writer.close();
    }

    public void writeTopicWordPros(String name) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(baseName + name + ".phi"));
        for (int t = 0; t < lflda.numTopics; t++) {
            for (int w = 0; w < lflda.vocabularySize; w++) {
                double pro = lflda.lambda * lflda.expDotProductValues[t][w] / lflda.sumExpValues[t] + (1 - lflda.lambda)
                        * (lflda.topicWordCountLDA[t][w] + lflda.beta) / (lflda.sumTopicWordCountLDA[t] + lflda.betaSum);
                writer.write(pro + " ");
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void writeDocTopicPros(String name) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(baseName + name + ".document-topics"));

        for (int i = 0; i < lflda.numDocuments; i++) {
            for (int j = 0; j < lflda.numTopics; j++) {
                double pro = (lflda.docTopicCount[i][j] + lflda.alpha[j]) / (lflda.sumDocTopicCount[i] + lflda.alphaSum);
                if (j == 0)
                    writer.write("" + pro);
                else
                    writer.write(" " + pro);
            }
            writer.write("\n");
        }
        writer.close();
    }

    public void write(int iteration) throws IOException {
        String name = String.format("iteration-%03d", iteration);
        this.lflda.topWords = 10;
        writeTopTopicalWords(name);
        writeDocTopicPros(name);
        if (iteration == this.lflda.numIterations) {
            writeParameters();
            writeTopicAssignments(name);
            writeTopicWordPros(name);
            this.lflda.topWords = 500;
            writeTopTopicalWords(name);
        }
    }
}
