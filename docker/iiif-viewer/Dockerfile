# One image that serves the pipeline output for viewing:
#   :8182  Cantaloupe, the IIIF Image API server for the original scans (mounted at /images)
#   :8000  nginx, serving the manifests (mounted at /srv/iiif) and a Mirador viewer at /viewer.html
#          (Ctrl+F searches the page text, see find.js)
FROM eclipse-temurin:25-jre-noble

ARG CANTALOUPE_VERSION=5.0.7
ARG CANTALOUPE_SHA256=741ddee3f5284b179cd2555dc545a8afcb02534f03896cfa9235249fd66b0665
# mirador-fixes.js patches performance bugs in this version; recheck it when upgrading
ARG MIRADOR_VERSION=4.3.0
ARG MIRADOR_SHA256=38e6aefcf445ce7348838a954f5b76220b5ef03e6772a5998f77c32e541ddd84

# nginx for the manifests and viewer; opj_decompress lets Cantaloupe read JPEG2000
RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx libopenjp2-tools unzip curl \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default

RUN curl -fsSL -o /tmp/cantaloupe.zip \
        "https://github.com/cantaloupe-project/cantaloupe/releases/download/v${CANTALOUPE_VERSION}/cantaloupe-${CANTALOUPE_VERSION}.zip" \
    && echo "${CANTALOUPE_SHA256}  /tmp/cantaloupe.zip" | sha256sum -c - \
    && unzip -q /tmp/cantaloupe.zip -d /opt \
    && mv /opt/cantaloupe-${CANTALOUPE_VERSION} /opt/cantaloupe \
    && mv /opt/cantaloupe/cantaloupe-${CANTALOUPE_VERSION}.jar /opt/cantaloupe/cantaloupe.jar \
    && rm /tmp/cantaloupe.zip \
    && mkdir -p /srv/viewer /srv/iiif /images /var/cache/cantaloupe \
    && curl -fsSL -o /srv/viewer/mirador.min.js \
        "https://unpkg.com/mirador@${MIRADOR_VERSION}/dist/mirador.min.js" \
    && echo "${MIRADOR_SHA256}  /srv/viewer/mirador.min.js" | sha256sum -c -

# Start from the shipped sample so every key exists, then override what this setup needs.
# JP2 goes to OpenJPEG (the automatic strategy picks Kakadu, which is licensed and not installed);
# TIFF and PNG fall back to Java2dProcessor.
RUN cd /opt/cantaloupe && cp cantaloupe.properties.sample cantaloupe.properties && sed -i \
    -e 's|^FilesystemSource.BasicLookupStrategy.path_prefix =.*|FilesystemSource.BasicLookupStrategy.path_prefix = /images/|' \
    -e 's|^processor.selection_strategy =.*|processor.selection_strategy = ManualSelectionStrategy|' \
    -e 's|^processor.ManualSelectionStrategy.jp2 =.*|processor.ManualSelectionStrategy.jp2 = OpenJpegProcessor|' \
    -e 's|^OpenJpegProcessor.path_to_binaries =.*|OpenJpegProcessor.path_to_binaries = /usr/bin|' \
    -e 's|^cache.server.derivative.enabled =.*|cache.server.derivative.enabled = true|' \
    -e 's|^cache.server.derivative =.*|cache.server.derivative = FilesystemCache|' \
    -e 's|^FilesystemCache.pathname =.*|FilesystemCache.pathname = /var/cache/cantaloupe|' \
    -e 's|^log.application.level =.*|log.application.level = info|' \
    cantaloupe.properties \
    && test "$(grep -cE '^(FilesystemSource.BasicLookupStrategy.path_prefix = /images/|processor.selection_strategy = ManualSelectionStrategy|processor.ManualSelectionStrategy.jp2 = OpenJpegProcessor|OpenJpegProcessor.path_to_binaries = /usr/bin|cache.server.derivative.enabled = true|cache.server.derivative = FilesystemCache|log.application.level = info)$' cantaloupe.properties)" = 7

COPY nginx.conf /etc/nginx/conf.d/iiif.conf
COPY viewer.html mirador-fixes.js find.js /srv/viewer/
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Heap for Cantaloupe; large untiled scans need several GB to decode
ENV JAVA_OPTS="-Xmx6g"
EXPOSE 8000 8182
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
