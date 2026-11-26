#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <sstream>

// =====================================================================
//  NSL Error & Diagnostics
//  SourceLocation — file, line, column
//  Diagnostic     — severity + location + message
//  DiagnosticEngine — collects and formats diagnostics
// =====================================================================

namespace NSL
{
    // ========================= SOURCE LOCATION =======================

    struct SourceLocation
    {
        std::string file;
        uint32_t    line   = 0;
        uint32_t    column = 0;

        std::string ToString() const
        {
            std::string result;
            if (!file.empty()) result += file + ":";
            result += std::to_string(line);
            if (column > 0) result += ":" + std::to_string(column);
            return result;
        }

        bool IsValid() const { return line > 0; }
    };

    // ========================= DIAGNOSTIC ============================

    struct Diagnostic
    {
        enum class Level : uint8_t { NOTE, WARNING, ERROR };

        Level          level    = Level::ERROR;
        SourceLocation location;
        std::string    message;

        std::string Format() const
        {
            std::string prefix;
            switch (level)
            {
                case Level::NOTE:    prefix = "note";    break;
                case Level::WARNING: prefix = "warning"; break;
                case Level::ERROR:   prefix = "error";   break;
            }

            if (location.IsValid())
                return location.ToString() + ": " + prefix + ": " + message;

            return prefix + ": " + message;
        }
    };

    // ========================= DIAGNOSTIC ENGINE =====================

    class DiagnosticEngine
    {
    public:
        void SetFile(const std::string& file) { m_file = file; }

        // ==================== Report ====================

        void ReportError(uint32_t line, uint32_t column, const std::string& message)
        {
            m_diagnostics.push_back({
                Diagnostic::Level::ERROR,
                { m_file, line, column },
                message
            });
            ++m_errorCount;
        }

        void ReportError(const SourceLocation& loc, const std::string& message)
        {
            m_diagnostics.push_back({
                Diagnostic::Level::ERROR,
                { loc.file.empty() ? m_file : loc.file, loc.line, loc.column },
                message
            });
            ++m_errorCount;
        }

        void ReportWarning(uint32_t line, uint32_t column, const std::string& message)
        {
            m_diagnostics.push_back({
                Diagnostic::Level::WARNING,
                { m_file, line, column },
                message
            });
            ++m_warningCount;
        }

        void ReportWarning(const SourceLocation& loc, const std::string& message)
        {
            m_diagnostics.push_back({
                Diagnostic::Level::WARNING,
                { loc.file.empty() ? m_file : loc.file, loc.line, loc.column },
                message
            });
            ++m_warningCount;
        }

        void ReportNote(uint32_t line, uint32_t column, const std::string& message)
        {
            m_diagnostics.push_back({
                Diagnostic::Level::NOTE,
                { m_file, line, column },
                message
            });
        }

        void ReportNote(const SourceLocation& loc, const std::string& message)
        {
            m_diagnostics.push_back({
                Diagnostic::Level::NOTE,
                { loc.file.empty() ? m_file : loc.file, loc.line, loc.column },
                message
            });
        }

        const std::string& GetFile() const { return m_file; }

        // ==================== Queries ====================

        bool     HasErrors()      const { return m_errorCount > 0; }
        uint32_t ErrorCount()     const { return m_errorCount; }
        uint32_t WarningCount()   const { return m_warningCount; }

        const std::vector<Diagnostic>& GetDiagnostics() const { return m_diagnostics; }

        // ==================== Formatting ====================

        std::string FormatAll() const
        {
            std::ostringstream ss;
            for (const auto& d : m_diagnostics)
                ss << d.Format() << "\n";

            if (m_errorCount > 0 || m_warningCount > 0)
                ss << m_errorCount << " error(s), " << m_warningCount << " warning(s).\n";

            return ss.str();
        }

        void PrintAll() const
        {
            for (const auto& d : m_diagnostics)
            {
                if (d.level == Diagnostic::Level::ERROR ||
                    d.level == Diagnostic::Level::WARNING)
                {
                    // Write to stderr
                    fprintf(stderr, "%s\n", d.Format().c_str());
                }
                else
                {
                    fprintf(stdout, "%s\n", d.Format().c_str());
                }
            }
        }

        void Clear()
        {
            m_diagnostics.clear();
            m_errorCount   = 0;
            m_warningCount = 0;
        }

    private:
        std::string             m_file;
        std::vector<Diagnostic> m_diagnostics;
        uint32_t                m_errorCount   = 0;
        uint32_t                m_warningCount = 0;
    };

} // namespace NSL
