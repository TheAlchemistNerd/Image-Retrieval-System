package com.retrieval.features.rules;

import com.retrieval.features.extractor.DJLExtractor;
import com.retrieval.features.extractor.Extractable;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

public class ResolutionRule implements ExtractorRule {
    private static final int MIN_RES = Integer.parseInt(System.getenv().getOrDefault("DEEP_LEARNING_MIN_RESOLUTION", "224"));

    @Override
    public Extractable apply(File file) {
        try {
            BufferedImage img = ImageIO.read(file);
            if (img != null && img.getWidth() >= MIN_RES && img.getHeight() >= MIN_RES) {
                return new DJLExtractor(); // Deep learning preferred for high-res
            }
        } catch (Exception ignored) {

        }
        return null;
    }
}
