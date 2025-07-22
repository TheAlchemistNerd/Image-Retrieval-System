package com.retrieval.features.extractor;

import boofcv.abst.feature.detdesc.DetectDescribePoint;
import boofcv.factory.feature.detdesc.FactoryDetectDescribe;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.feature.TupleDesc_F64;
import boofcv.struct.image.GrayF32;
import com.retrieval.utils.FeatureUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BoofCVExtractor implements Extractable {
    private static final Logger log = LoggerFactory.getLogger(BoofCVExtractor.class);

    @Override
    public double[] extract(String imagePath) {
        if (imagePath == null || imagePath.trim().isEmpty()) {
            throw new IllegalArgumentException("Image path cannot be null or empty");
        }

        try {
            GrayF32 image = UtilImageIO.loadImage(imagePath, GrayF32.class);
            if (image == null) {
                log.error("BoofCVExtractor: Could not load image {}", imagePath);
                return new double[0];
            }

            DetectDescribePoint<GrayF32, TupleDesc_F64> surf =
                    FactoryDetectDescribe.surfStable(null, null, null, GrayF32.class);
            surf.detect(image);

            int numberOfFeatures = surf.getNumberOfFeatures();
            if (numberOfFeatures == 0) {
                log.warn("BoofCVExtractor: No features found in {}", imagePath);
                return new double[0];
            }


            int descriptorLength = surf.getDescription(0).data.length;
            double[] averageDescriptor = new double[descriptorLength];

            // Process each descriptor
            for (int i = 0; i < numberOfFeatures; i++) {

                double[] descriptor = surf.getDescription(i).data.clone();

                // Normalize individual descriptor before averaging
                FeatureUtils.normalize(descriptor);

                // Add to running average
                for (int j = 0; j < descriptorLength; j++) {
                    averageDescriptor[j] += descriptor[j];
                }
            }

            // Complete the averaging
            for (int i = 0; i < averageDescriptor.length; i++) {
                averageDescriptor[i] /= numberOfFeatures;
            }

            // Final normalization of the averaged descriptor
            FeatureUtils.normalize(averageDescriptor);

            log.debug("BoofCVExtractor: Extracted {} features from {}, final descriptor length: {}",
                    numberOfFeatures, imagePath, descriptorLength);

            return averageDescriptor;

        } catch (Exception e) {
            log.error("Error extracting BoofCV features for {}", imagePath, e);
            return new double[0];
        }
    }
}
