#pragma once

#include "texture_types.h"
#include "texture_desc.h"
#include "../types/northstar_types.h"
#include <cstring>
#include <vector>

namespace Northstar
{
    // ============ Texture ============
    class Texture
    {
    public:
        NativeHandle       GetNativeHandle()                           const { return m_nativeHandle;      }
        // ==================== DIMENSION GETTERS ====================
        UINT32             GetWidth()                                  const { return m_width;             }
        UINT32             GetHeight()                                 const { return m_height;            }
        UINT32             GetDepth()                                  const { return m_depth;             }
        UINT32             GetMipLevels()                              const { return m_mipLevels;         }
        UINT32             GetArrayLayers()                            const { return m_arrayLayers;       }

        // ==================== FORMAT & TYPE GETTERS ====================
        TextureFormat      GetFormat()                                 const { return m_format;            }
        TextureType        GetType()                                   const { return m_type;              }
        TextureUsage       GetUsage()                                  const { return m_usage;             }
        TextureDataSource  GetDataSource()                             const { return m_dataSource;        }
        BOOL               AllowsSharedHandle()                        const { return m_allowSharedHandle; }

        // NSL DataTypeID / ResourceKind for PROPERTY texture Get/Add slots
        DataTypeID         GetNSLDataTypeID()                          const { return TextureTypeToDataTypeID(m_type, m_usage); }
        ResourceKind       GetNSLResourceKind()                        const { return TextureTypeToResourceKind(m_type, m_usage); }

        // ==================== NSL INTEGRATION ====================
        BOOL MatchesNSCTexture(const NSCTexture& nscTex) const
        {
            return GetNSLDataTypeID() == nscTex.typeID;
        }

    private:
        friend class NorthstarEngine;
        friend class NorthstarCore;
        friend class CoreTextureManager;

        Texture() = default;

        void SetNativeHandle(NativeHandle handle)
        {
            m_nativeHandle = handle;
            if (handle != 0)
            {
                m_pendingInitialData.clear();
                m_pendingInitialData.shrink_to_fit();
            }
        }
        void SetDimensions(UINT32 width, UINT32 height, UINT32 depth = 1, UINT32 mips = 1, UINT32 layers = 1, BOOL autoMipChain = false)
        {
            m_width        = width;
            m_height       = height;
            m_depth        = depth;
            m_mipLevels    = mips;
            m_arrayLayers  = layers;
            m_autoMipChain = autoMipChain;
        }
        void SetFormatAndType(
            TextureFormat     format,
            TextureType       type,
            TextureUsage      usage,
            TextureDataSource dataSource        = TextureDataSource::EMPTY,
            BOOL              allowSharedHandle = false)
        {
            m_format            = format;
            m_type              = type;
            m_usage             = usage;
            m_dataSource        = dataSource;
            m_allowSharedHandle = allowSharedHandle;
        }
        void SetPendingInitialData(const void* data, SIZE_T byteSize)
        {
            if (data != nullptr && byteSize > 0u)
            {
                m_pendingInitialData.resize(byteSize);
                std::memcpy(m_pendingInitialData.data(), data, byteSize);
            }
            else
            {
                m_pendingInitialData.clear();
                m_pendingInitialData.shrink_to_fit();
            }
        }
        const BYTE* GetPendingInitialData()     const { return m_pendingInitialData.empty() ? nullptr : m_pendingInitialData.data(); }
        SIZE_T      GetPendingInitialDataSize() const { return m_pendingInitialData.size(); }
        BOOL        UsesAutoMipChain()          const { return m_autoMipChain; }

        NativeHandle       m_nativeHandle      = 0;

        // Dimensions
        UINT32             m_width             = 1;
        UINT32             m_height            = 1;
        UINT32             m_depth             = 1;
        UINT32             m_mipLevels         = 1;
        UINT32             m_arrayLayers       = 1;
        BOOL               m_autoMipChain      = false;

        // Format & Type
        TextureFormat      m_format            = TextureFormat::RGBA8_UNORM;
        TextureType        m_type              = TextureType::TEXTURE_2D;
        TextureUsage       m_usage             = TextureUsage::SAMPLED;
        TextureDataSource  m_dataSource        = TextureDataSource::EMPTY;
        BOOL               m_allowSharedHandle = false;
        std::vector<BYTE>  m_pendingInitialData;
    };
}
