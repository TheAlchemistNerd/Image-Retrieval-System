package com.retrieval.features.extractor;

import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import com.retrieval.utils.FeatureUtils;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Feature extractor using DL4J's ResNet50 ZooModel.
 */
public class DL4JExtractor implements Extractable {

    private static final int IMG_WIDTH = 224;
    private static final int IMG_HEIGHT = 224;
    private static final int IMG_CHANNELS = 3;

    private static final Logger logger = Logger.getLogger(DL4JExtractor.class.getName());

    private static class LazyHolder {
        static final ComputationGraph MODEL = loadModel();
    }

    private static ComputationGraph loadModel() {
        try {
            ZooModel zooModel = ResNet50.builder().numClasses(1000).build();
            ComputationGraph model = (ComputationGraph) zooModel.initPretrained();
            logger.info("DL4J ResNet50 model loaded.");
            return model;
        } catch (Exception e) {
            logger.severe("Failed to load ResNet50 model: " + e.getMessage());
            return null;
        }
    }

    @Override
    public double[] extract(String imagePath) {
        ComputationGraph model = LazyHolder.MODEL;
        if (model == null) return new double[0];

        try {
            // Fix: Use ResizeImageTransform instead of CenterCropImageTransform
            ImageTransform transform = new ResizeImageTransform(IMG_WIDTH, IMG_HEIGHT);
            NativeImageLoader loader = new NativeImageLoader(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, transform);
            INDArray image = loader.asMatrix(new File(imagePath));
            new VGG16ImagePreProcessor().transform(image);

            // Maintain insertion order of activations
            Map<String, INDArray> activations = new LinkedHashMap<>(model.feedForward(image, false));

            INDArray features = activations.get("avg_pool");
            if (features == null) {
                logger.warning("'avg_pool' not found, falling back to last hidden layer.");
                String fallbackLayer = getLastHiddenLayer(activations);
                if (fallbackLayer == null) return new double[0];

                features = activations.get(fallbackLayer);
                logger.info("Using fallback layer: " + fallbackLayer);

                if (features.rank() == 4) {
                    // Global average pooling over H and W
                    features = features.mean(2).mean(3);
                }
            }

            double[] vector = features.ravel().toDoubleVector();
            FeatureUtils.normalize(vector);
            return vector;

        } catch (Exception e) {
            logger.severe("Error extracting features from image " + imagePath + ": " + e.getMessage());
            return new double[0];
        }
    }

    private String getLastHiddenLayer(Map<String, INDArray> activations) {
        return activations.keySet().stream()
                .filter(k -> !k.equalsIgnoreCase("output"))
                .reduce((first, second) -> second)
                .orElse(null);
    }
}