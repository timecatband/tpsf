//
//  tpfinstrumentExtensionParameterAddresses.h
//  tpfinstrumentExtension
//
//  Created by Robert Carr on 5/5/24.
//

#pragma once

#include <AudioToolbox/AUParameters.h>

#ifdef __cplusplus
namespace tpfinstrumentExtensionParameterAddress {
#endif

typedef NS_ENUM(AUParameterAddress, tpfinstrumentExtensionParameterAddress) {
    gain = 0
};

#ifdef __cplusplus
}
#endif
