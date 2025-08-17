#include <iostream>
#include <memory>
#include <string>

#include <gst/gst.h>

#include "NvDrmRenderer.h"
#include "tegra_drm_nvdc.h"

#include "argus_capture.h"



int main(int argc, char** argv) {
    struct drm_tegra_hdr_metadata_smpte_2086 drm_metadata;
    NvDrmRenderer *hdmi = NvDrmRenderer::createDrmRenderer("renderer0", 1920, 1080, 0, 0,
            /*connector*/ 0, /*crtc*/ 0, /*plane_id*/ 0, drm_metadata, true);

    ArgusCapture ac;
    ac.run();
    while(1);
    
    return 0;
}
