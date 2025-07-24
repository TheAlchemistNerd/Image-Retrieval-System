package com.retrieval.features.extractor;

import ai.djl.Application;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.*;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.*;
import com.retrieval.utils.FeatureUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Paths;

/**
 * Feature extractor using DJL and a pre-trained ResNet50 model.
 */
public class DJLExtractor implements Extractable, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DJLExtractor.class);

    private static final int IMG_SIZE = 224;
    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};

    private ZooModel<Image, float[]> model;
    private Predictor<Image, float[]> predictor;

    public DJLExtractor() {
        try {
            Criteria<Image, float[]> criteria = Criteria.builder()
                    .setTypes(Image.class, float[].class)
                    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                    .optFilter("backbone", "resnet50")
                    .optEngine("PyTorch")
                    .optTranslator(new FeatureExtractorTranslator())
                    .optProgress(new ai.djl.training.util.ProgressBar())
                    .build();

            this.model = ModelZoo.loadModel(criteria);
            this.predictor = model.newPredictor();
            log.info("DJL ResNet50 model loaded successfully.");
        } catch (Exception e) {
            log.error("Failed to load DJL ResNet50 model", e);
        }
    }

    @Override
    public double[] extract(String imagePath) {
        if (predictor == null) {
            log.error("DJLExtractor: Model is not loaded.");
            return new double[0];
        }

        try {
            Image image = ImageFactory.getInstance().fromFile(Paths.get(imagePath));
            float[] features = predictor.predict(image);

            double[] featureVector = new double[features.length];
            for (int i = 0; i < features.length; i++) {
                featureVector[i] = features[i];
            }

            FeatureUtils.normalize(featureVector);
            log.debug("DJLExtractor: Extracted {} features from {}", featureVector.length, imagePath);
            return featureVector;

        } catch (IOException | TranslateException e) {
            log.error("Error extracting features from image: {}", imagePath, e);
            return new double[0];
        }
    }

    private static class FeatureExtractorTranslator implements Translator<Image, float[]> {

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            // Convert Image to NDList first
            NDList array = new NDList(input.toNDArray(ctx.getNDManager()));

            Pipeline pipeline = new Pipeline();
            pipeline.add(new Resize(256))
                    .add(new CenterCrop(IMG_SIZE, IMG_SIZE))
                    .add(new ToTensor())
                    .add(new Normalize(IMAGENET_MEAN, IMAGENET_STD));

            // Apply transforms to the NDList
            array = pipeline.transform(array);
            return array;
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray features = list.singletonOrThrow();
            if (features.getShape().dimension() > 2) {
                features = features.mean(new int[]{2, 3}); // Global average pooling over H & W
            }
            return features.toFloatArray();
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }

    @Override
    public void close() {
        if (predictor != null) predictor.close();
        if (model != null) model.close();
    }
}