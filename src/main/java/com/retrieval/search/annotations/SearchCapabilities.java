package com.retrieval.search.annotations;

import java.lang.annotation.*;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface SearchCapabilities {
    boolean insertable() default false;
    boolean buildable() default true;
    boolean searchable() default true;
}

