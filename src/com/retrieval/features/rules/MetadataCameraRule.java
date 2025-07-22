package com.retrieval.features.rules;

import com.drew.imaging.ImageMetadataReader;
import com.drew.metadata.Metadata;
import com.drew.metadata.exif.ExifIFD0Directory;
import com.retrieval.features.extractor.DL4JExtractor;
import com.retrieval.features.extractor.Extractable;

import java.io.File;

public class MetadataCameraRule implements ExtractorRule {
    @Override
    public Extractable apply(File file) {
        try {
            Metadata metadata = ImageMetadataReader.readMetadata(file);
            ExifIFD0Directory dir = metadata.getFirstDirectoryOfType(ExifIFD0Directory.class);
            if (dir != null && dir.containsTag(ExifIFD0Directory.TAG_MODEL)) {
                String model = dir.getString(ExifIFD0Directory.TAG_MODEL).toLowerCase();
                if(model.contains("canon") || model.contains("nikon") || model.contains("sony")) {
                    return new DL4JExtractor(); //deep learning preferred for DSLR
                }
            }
        } catch (Exception e) {
            // Safe fallback: do nothing
        }
        return null;
    }
}
