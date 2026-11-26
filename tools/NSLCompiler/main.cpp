#include "core/nsl_error.h"
#include "core/nsl_preprocessor.h"
#include "core/nsl_file_parser.h"
#include "core/nsl_library_resolver.h"
#include "property/nsl_property_parser.h"
#include "property/nsl_binding_resolver.h"
#include "stage_io/nsl_stage_io_parser.h"
#include "transpiler/lexer/nsl_lexer.h"
#include "transpiler/parser/nsl_parser.h"
#include "transpiler/sema/nsl_sema.h"
#include "transpiler/backend/hlsl/nsl_hlsl_backend.h"
#include "transpiler/backend/hlsl/nsl_hlsl_stage_io.h"
#include "aligner/hlsl/nsl_hlsl_aligner.h"
#include "writer/nsl_nsc_writer.h"
#include "NSLCommon/nsl_common_types.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>

static const char* kVersion = "3.2.0";

// Inject STAGE_INPUT / STAGE_OUTPUT structs from stage layouts into the AST.
static bool InjectStageIOStructs(
    NSL::Program& ast,
    const NSL::ParsedStageIO& stageIO,
    NSL::DiagnosticEngine& diag)
{
    auto hasStruct = [&](const char* name) -> bool
    {
        for (const auto& s : ast.structs)
            if (s.name == name) return true;
        return false;
    };

    auto makeStruct = [](const char* name, const std::vector<NSL::StageIOEntry>& entries) -> NSL::StructDecl
    {
        NSL::StructDecl sd;
        sd.name = name;
        for (const auto& e : entries)
        {
            NSL::StructField f;
            const NSL::DataTypeInfo* info = NSL::FindDataTypeByID(e.typeID);
            f.typeName = info ? info->nslName : "Float";
            f.name     = e.name;
            f.semantic = e.name; // NSL catalog id; backend maps at emit
            sd.fields.push_back(std::move(f));
        }
        return sd;
    };

    if (hasStruct("STAGE_INPUT") || hasStruct("STAGE_OUTPUT"))
    {
        diag.ReportError(0, 0,
            "STAGE_INPUT and STAGE_OUTPUT are compiler-reserved names; "
            "declare fields through INPUT_LAYOUT / OUTPUT_LAYOUT.");
        return false;
    }

    if (!stageIO.inputs.empty())
        ast.structs.insert(ast.structs.begin(), makeStruct("STAGE_INPUT", stageIO.inputs));

    if (!stageIO.outputs.empty() || ast.stage == NSL::ShaderStage::PIXEL)
    {
        size_t insertAt = 0;
        if (!ast.structs.empty() && ast.structs.front().name == "STAGE_INPUT")
            insertAt = 1;
        ast.structs.insert(ast.structs.begin() + static_cast<std::ptrdiff_t>(insertAt),
                           makeStruct("STAGE_OUTPUT", stageIO.outputs));
    }
    return true;
}

// Extract the value inside the first "(...)" of an attribute's raw args.
// e.g. args = "domain(tri)" -> "tri"; string literal quotes are already stripped.
static std::string AttributeValue(const NSL::Attribute& attr)
{
    size_t open  = attr.args.find('(');
    size_t close = attr.args.rfind(')');
    if (open == std::string::npos || close == std::string::npos || close <= open + 1)
        return {};
    std::string v = attr.args.substr(open + 1, close - open - 1);
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"')
        v = v.substr(1, v.size() - 2);
    return v;
}

// Inject the compiler-generated PATCH_CONSTANT_DATA struct for tessellation
// stages. Tess-factor field layout is derived from the [domain("...")] attribute
// on Main (HULL: control-point Main; DOMAIN: Main). Returns silently if the
// domain is missing/unknown — sema reports the diagnostic.
static bool InjectPatchConstantData(
    NSL::Program& ast,
    NSL::DiagnosticEngine& diag)
{
    if (ast.stage != NSL::ShaderStage::HULL && ast.stage != NSL::ShaderStage::DOMAIN)
        return true;

    for (const auto& s : ast.structs)
    {
        if (s.name == "PATCH_CONSTANT_DATA")
        {
            diag.ReportError(s.location,
                "PATCH_CONSTANT_DATA is compiler-generated and cannot be declared manually.");
            return false;
        }
    }

    std::string domain;
    for (const auto& fn : ast.functions)
    {
        if (fn.name != "Main") continue;
        for (const auto& attr : fn.attributes)
            if (attr.name == "domain") { domain = AttributeValue(attr); break; }
        break;
    }
    if (domain.empty()) return true;

    auto makeField = [](const char* type, const char* name,
                        const char* arraySize, const char* semantic) -> NSL::StructField
    {
        NSL::StructField f;
        f.typeName      = type;
        f.name          = name;
        f.arraySizeExpr = arraySize;
        f.semantic      = semantic;
        return f;
    };

    NSL::StructDecl sd;
    sd.name = "PATCH_CONSTANT_DATA";

    if (domain == "tri")
    {
        sd.fields.push_back(makeField("Float", "EdgeFactors", "3", "TESS_FACTOR"));
        sd.fields.push_back(makeField("Float", "InsideFactor", "", "INSIDE_TESS_FACTOR"));
    }
    else if (domain == "quad")
    {
        sd.fields.push_back(makeField("Float", "EdgeFactors", "4", "TESS_FACTOR"));
        sd.fields.push_back(makeField("Float", "InsideFactors", "2", "INSIDE_TESS_FACTOR"));
    }
    else if (domain == "isoline")
    {
        sd.fields.push_back(makeField("Float", "EdgeFactors", "2", "TESS_FACTOR"));
    }
    else
    {
        return true; // unknown domain — sema will report it
    }

    ast.structs.insert(ast.structs.begin(), std::move(sd));
    return true;
}

// =====================================================================
//  Coverage / Discard Call-Graph Analyzer
//  Detects discard / clip() across Main and all transitively called
//  helper / library functions so NSC_FLAG_USES_DISCARD is accurate.
// =====================================================================

struct CoverageCallGraphAnalyzer
{
    static void InspectExpr(
        const NSL::Expression*           expr,
        bool&                            outUsesDiscard,
        std::unordered_set<std::string>& outCallees)
    {
        if (!expr) return;

        if (const auto* ce = dynamic_cast<const NSL::CallExpr*>(expr))
        {
            if (const auto* id = dynamic_cast<const NSL::IdentifierExpr*>(ce->callee.get()))
            {
                if (id->name == "clip" || id->name == "Clip")
                    outUsesDiscard = true;
                else
                    outCallees.insert(id->name);
            }
            InspectExpr(ce->callee.get(), outUsesDiscard, outCallees);
            for (const auto& arg : ce->args)
                InspectExpr(arg.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* be = dynamic_cast<const NSL::BinaryExpr*>(expr))
        {
            InspectExpr(be->left.get(), outUsesDiscard, outCallees);
            InspectExpr(be->right.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* ue = dynamic_cast<const NSL::UnaryExpr*>(expr))
        {
            InspectExpr(ue->operand.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* pe = dynamic_cast<const NSL::ParenExpr*>(expr))
        {
            InspectExpr(pe->inner.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* te = dynamic_cast<const NSL::TernaryExpr*>(expr))
        {
            InspectExpr(te->condition.get(), outUsesDiscard, outCallees);
            InspectExpr(te->thenExpr.get(), outUsesDiscard, outCallees);
            InspectExpr(te->elseExpr.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* ie = dynamic_cast<const NSL::IndexExpr*>(expr))
        {
            InspectExpr(ie->object.get(), outUsesDiscard, outCallees);
            InspectExpr(ie->index.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* ma = dynamic_cast<const NSL::MemberAccessExpr*>(expr))
        {
            InspectExpr(ma->object.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* sw = dynamic_cast<const NSL::SwizzleExpr*>(expr))
        {
            InspectExpr(sw->object.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* le = dynamic_cast<const NSL::LengthExpr*>(expr))
        {
            InspectExpr(le->object.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* cast = dynamic_cast<const NSL::CastExpr*>(expr))
        {
            for (const auto& arg : cast->args)
                InspectExpr(arg.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* init = dynamic_cast<const NSL::InitializerListExpr*>(expr))
        {
            for (const auto& el : init->elements)
                InspectExpr(el.get(), outUsesDiscard, outCallees);
        }
    }

    static void InspectStmt(
        const NSL::Statement*            stmt,
        bool&                            outUsesDiscard,
        std::unordered_set<std::string>& outCallees)
    {
        if (!stmt) return;

        if (dynamic_cast<const NSL::DiscardStmt*>(stmt) != nullptr)
        {
            outUsesDiscard = true;
        }
        else if (const auto* es = dynamic_cast<const NSL::ExprStmt*>(stmt))
        {
            InspectExpr(es->expr.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* rs = dynamic_cast<const NSL::ReturnStmt*>(stmt))
        {
            InspectExpr(rs->value.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* vd = dynamic_cast<const NSL::VarDeclStmt*>(stmt))
        {
            InspectExpr(vd->initializer.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* blk = dynamic_cast<const NSL::BlockStmt*>(stmt))
        {
            for (const auto& child : blk->statements)
                InspectStmt(child.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* ifs = dynamic_cast<const NSL::IfStmt*>(stmt))
        {
            InspectExpr(ifs->condition.get(), outUsesDiscard, outCallees);
            InspectStmt(ifs->thenBranch.get(), outUsesDiscard, outCallees);
            InspectStmt(ifs->elseBranch.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* fs = dynamic_cast<const NSL::ForStmt*>(stmt))
        {
            InspectStmt(fs->init.get(), outUsesDiscard, outCallees);
            InspectExpr(fs->condition.get(), outUsesDiscard, outCallees);
            InspectExpr(fs->increment.get(), outUsesDiscard, outCallees);
            InspectStmt(fs->body.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* ws = dynamic_cast<const NSL::WhileStmt*>(stmt))
        {
            InspectExpr(ws->condition.get(), outUsesDiscard, outCallees);
            InspectStmt(ws->body.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* dws = dynamic_cast<const NSL::DoWhileStmt*>(stmt))
        {
            InspectStmt(dws->body.get(), outUsesDiscard, outCallees);
            InspectExpr(dws->condition.get(), outUsesDiscard, outCallees);
        }
        else if (const auto* sw = dynamic_cast<const NSL::SwitchStmt*>(stmt))
        {
            InspectExpr(sw->condition.get(), outUsesDiscard, outCallees);
            for (const auto& sc : sw->cases)
            {
                InspectExpr(sc.value.get(), outUsesDiscard, outCallees);
                for (const auto& child : sc.statements)
                    InspectStmt(child.get(), outUsesDiscard, outCallees);
            }
        }
    }

    static uint16_t ComputeFlags(const NSL::Program& ast)
    {
        struct FnSummary
        {
            bool                            usesDiscard = false;
            std::unordered_set<std::string> callees;
        };

        std::unordered_map<std::string, FnSummary> summaries;
        for (const auto& fn : ast.functions)
        {
            FnSummary s;
            for (const auto& stmt : fn.body)
                InspectStmt(stmt.get(), s.usesDiscard, s.callees);
            FnSummary& dst = summaries[fn.name];
            dst.usesDiscard = dst.usesDiscard || s.usesDiscard;
            dst.callees.insert(s.callees.begin(), s.callees.end());
        }

        std::vector<std::string> worklist;
        std::unordered_set<std::string> visited;
        if (summaries.count("Main"))
        {
            worklist.push_back("Main");
            visited.insert("Main");
        }
        else
        {
            for (const auto& fn : ast.functions)
            {
                worklist.push_back(fn.name);
                visited.insert(fn.name);
            }
        }

        bool reachableDiscard = false;
        while (!worklist.empty())
        {
            std::string curr = std::move(worklist.back());
            worklist.pop_back();

            auto it = summaries.find(curr);
            if (it == summaries.end()) continue;
            if (it->second.usesDiscard)
            {
                reachableDiscard = true;
                break;
            }
            for (const auto& callee : it->second.callees)
            {
                if (summaries.count(callee) && visited.insert(callee).second)
                    worklist.push_back(callee);
            }
        }

        uint16_t flags = NSL::NSC_FLAG_NONE;
        if (reachableDiscard)
            flags |= NSL::NSC_FLAG_USES_DISCARD;
        return flags;
    }
};

// =====================================================================
//  PrintUsage
// =====================================================================

static void PrintUsage()
{
    std::cout << "NSL Compiler v" << kVersion << "\n"
              << "Usage: NSLCompiler --input <file.nsl> --target <hlsl11|hlsl12> --output <file.nsc> [options]\n\n"
              << "Options:\n"
              << "  --input   <file>         Input .nsl source file\n"
              << "  --output  <file>         Output .nsc binary file path\n"
              << "  --target  <api>          Target API: hlsl11 or hlsl12 (default: hlsl11)\n"
              << "  -I, --include-dir <dir>  Add include search directory\n"
              << "  --verbose                Verbose pipeline output\n"
              << "  --emit-text <file>       Also emit human-readable .nslt debug output\n"
              << "  --version                Print version and exit\n\n"
              << "Example:\n"
              << "  NSLCompiler --input pbr.nsl --target hlsl12 --output pbr.nsc -I shaders\n";
}

// =====================================================================
//  main
// =====================================================================

int main(int argc, char* argv[])
{
    std::string                             inputFile;
    std::string                             outputFile;
    std::string                             targetAPI = "hlsl11";
    std::string                             textOutput;
    std::vector<std::filesystem::path>      includeDirs;
    bool                                    verbose   = false;

    // ==================== CLI argument parsing ====================
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--input")
        {
            if (i + 1 >= argc || argv[i + 1][0] == '-')
            {
                std::cerr << "error: Option '--input' requires a file path argument.\n";
                return 1;
            }
            inputFile = argv[++i];
        }
        else if (arg == "--output")
        {
            if (i + 1 >= argc || argv[i + 1][0] == '-')
            {
                std::cerr << "error: Option '--output' requires a file path argument.\n";
                return 1;
            }
            outputFile = argv[++i];
        }
        else if (arg == "-I" || arg == "--include-dir")
        {
            if (i + 1 >= argc || argv[i + 1][0] == '-')
            {
                std::cerr << "error: Option '" << arg << "' requires a directory path argument.\n";
                return 1;
            }
            includeDirs.push_back(argv[++i]);
        }
        else if (arg == "--target")
        {
            if (i + 1 >= argc || argv[i + 1][0] == '-')
            {
                std::cerr << "error: Option '--target' requires an API argument (hlsl11 or hlsl12).\n";
                return 1;
            }
            targetAPI = argv[++i];
        }
        else if (arg == "--emit-text")
        {
            if (i + 1 >= argc || argv[i + 1][0] == '-')
            {
                std::cerr << "error: Option '--emit-text' requires a file path argument.\n";
                return 1;
            }
            textOutput = argv[++i];
        }
        else if (arg == "--verbose")
        {
            verbose = true;
        }
        else if (arg == "--version")
        {
            std::cout << "NSL Compiler v" << kVersion << "\n";
            return 0;
        }
        else if (arg == "--help" || arg == "-h")
        {
            PrintUsage();
            return 0;
        }
        else
        {
            std::cerr << "error: Unrecognized command-line option '" << arg << "'. Use --help for usage.\n";
            return 1;
        }
    }

    if (inputFile.empty() || outputFile.empty())
    {
        PrintUsage();
        return 1;
    }

    if (targetAPI != "hlsl11" && targetAPI != "hlsl12")
    {
        std::cerr << "error: Target must be 'hlsl11' or 'hlsl12'.\n";
        return 1;
    }
    NSL::TargetAPI targetApiEnum = (targetAPI == "hlsl11") ? NSL::TargetAPI::HLSL11 : NSL::TargetAPI::HLSL12;

    // ==================== Read source file ====================
    std::ifstream inFile(inputFile, std::ios::ate | std::ios::binary);
    if (!inFile.is_open())
    {
        std::cerr << "error: Cannot open input file '" << inputFile << "'\n";
        return 1;
    }

    auto fileSize = inFile.tellg();
    if (fileSize > 10 * 1024 * 1024)
    {
        std::cerr << "error: Input file exceeds maximum size (10 MB).\n";
        return 1;
    }

    inFile.seekg(0);
    std::ostringstream oss;
    oss << inFile.rdbuf();
    std::string source = oss.str();
    if (source.size() >= 3 &&
        static_cast<unsigned char>(source[0]) == 0xEF &&
        static_cast<unsigned char>(source[1]) == 0xBB &&
        static_cast<unsigned char>(source[2]) == 0xBF)
    {
        source.erase(0, 3);
    }

    std::string baseDir  = std::filesystem::path(inputFile).parent_path().string();
    if (baseDir.empty()) baseDir = ".";
    std::string fileName = std::filesystem::path(inputFile).filename().string();

    NSL::DiagnosticEngine diag;
    diag.SetFile(fileName);

    if (verbose) std::cout << "[NSL] Input:  " << inputFile << "\n"
                           << "[NSL] Target: " << targetAPI << "\n"
                           << "[NSL] Output: " << outputFile << "\n";

    try
    {
        // ==================== 1. PREPROCESSOR ====================
        if (verbose) std::cout << "[NSL] Phase 1: Preprocessing (#include)...\n";

        NSL::Preprocessor preprocessor;

        // Auto-register parent directory of input file (e.g. shaders/ if input is shaders/StandardLit/StandardLitVS.nsl)
        std::filesystem::path parentDir = std::filesystem::path(baseDir).parent_path();
        if (!parentDir.empty() && std::filesystem::exists(parentDir))
        {
            preprocessor.AddIncludeDirectory(parentDir);
        }

        for (const auto& incDir : includeDirs)
        {
            preprocessor.AddIncludeDirectory(incDir);
        }

        std::string processedSource = preprocessor.Process(source, baseDir, fileName, diag);

        diag.PrintAll();
        if (diag.HasErrors()) return 2;

        // ==================== 2. FILE PARSER ====================
        if (verbose) std::cout << "[NSL] Phase 2: File structure parsing...\n";

        NSL::NSLFileParser fileParser(processedSource);
        NSL::NSLFileLayout layout;

        if (!fileParser.Parse(layout, diag))
        {
            diag.PrintAll();
            return 2;
        }

        if (layout.isLibrary && layout.stage == NSL::ShaderStage::NONE)
        {
            std::cerr << "error: Cannot compile library file '" << inputFile << "' as a standalone shader stage. Library shaders must be imported into a main shader via LIBRARY_DECLARATION.\n";
            return 1;
        }

        if (!layout.libraryDeclarations.empty())
        {
            if (verbose) std::cout << "[NSL] Phase 2b: Resolving " << layout.libraryDeclarations.size() << " library declaration(s)...\n";
            if (!NSL::LibraryResolver::Resolve(layout, baseDir, includeDirs, diag))
            {
                diag.PrintAll();
                return 2;
            }
        }

        if (verbose)
        {
            std::cout << "[NSL]   Stage: " << NSL::ShaderStageToString(layout.stage) << "\n"
                      << "[NSL]   " << layout.propertyBlocks.size() << " PROPERTY block(s)\n"
                      << "[NSL]   " << layout.stageIOBlocks.size() << " stage layout block(s)\n"
                      << "[NSL]   SHADER block: " << layout.shaderSource.size() << " chars\n";
        }

        // ==================== 3. PROPERTY PARSER ====================
        if (verbose) std::cout << "[NSL] Phase 3: PROPERTY block parsing...\n";

        NSL::PropertyParser propParser;
        NSL::ParsedProperties parsedProps;

        for (const auto& rawBlock : layout.propertyBlocks)
        {
            if (!propParser.ParseBlock(rawBlock.name, rawBlock.rawText, rawBlock.startLine,
                                       parsedProps, diag))
            {
                diag.PrintAll();
                return 2;
            }
        }

        if (verbose)
            std::cout << "[NSL]   " << parsedProps.groups.size() << " PROPERTY group(s) parsed\n";

        // ==================== 3b. STAGE I/O PARSER ====================
        if (verbose) std::cout << "[NSL] Phase 3b: INPUT/OUTPUT_LAYOUT parsing...\n";

        NSL::StageIOParser stageIOParser;
        NSL::ParsedStageIO parsedStageIO;

        for (const auto& rawIO : layout.stageIOBlocks)
        {
            NSL::StageIODirection dir = (rawIO.kind == NSL::StageIOBlockKind::INPUT)
                ? NSL::StageIODirection::INPUT
                : NSL::StageIODirection::OUTPUT;

            if (!stageIOParser.ParseBlock(dir, layout.stage, rawIO.rawText, rawIO.startLine,
                                          parsedStageIO, diag))
            {
                diag.PrintAll();
                return 2;
            }
        }

        if (!NSL::StageIOParser::Validate(layout.stage, parsedStageIO, diag))
        {
            diag.PrintAll();
            return 2;
        }

        if (verbose)
            std::cout << "[NSL]   INPUT entries: " << parsedStageIO.inputs.size()
                      << ", OUTPUT entries: " << parsedStageIO.outputs.size() << "\n";

        // ==================== 4. SLOT RESOLVER ====================
        if (verbose) std::cout << "[NSL] Phase 4: Slot assignment...\n";

        NSL::BindingResolver bindingResolver;
        if (!bindingResolver.Resolve(parsedProps, targetApiEnum, layout.stage, diag))
        {
            diag.PrintAll();
            return 2;
        }

        if (verbose)
        {
            for (const auto& g : parsedProps.groups)
            {
                std::cout << "[NSL]   " << NSL::ResourceKindToString(g.resourceKind)
                          << " '" << g.name << "' → set " << g.assignedSet << " binding " << g.assignedBinding << "\n";
            }
        }

        // ==================== 5. LEXER ====================
        if (verbose) std::cout << "[NSL] Phase 5: Lexing SHADER block...\n";

        const uint32_t lexerLineOffset = (layout.shaderStartLine > 0u) ? (layout.shaderStartLine - 1u) : 0u;
        NSL::Lexer lexer(layout.shaderSource, fileName, lexerLineOffset);
        auto tokens = lexer.Tokenize();

        if (verbose) std::cout << "[NSL]   " << tokens.size() << " tokens\n";

        // ==================== 6. PARSER ====================
        if (verbose) std::cout << "[NSL] Phase 6: Parsing AST...\n";

        NSL::Parser parser(tokens, parsedProps, fileName);
        NSL::Program ast = parser.Parse(diag);

        diag.PrintAll();
        if (diag.HasErrors()) return 2;

        // Set shader stage
        ast.stage = layout.stage;
        for (auto& fn : ast.functions)
            if (fn.name == "Main") fn.stage = ast.stage;

        // Inject STAGE_INPUT / STAGE_OUTPUT from layouts (before sema)
        if (!InjectStageIOStructs(ast, parsedStageIO, diag))
        {
            diag.PrintAll();
            return 2;
        }

        // Inject PATCH_CONSTANT_DATA for tessellation stages (before sema)
        if (!InjectPatchConstantData(ast, diag))
        {
            diag.PrintAll();
            return 2;
        }

        if (verbose)
        {
            std::cout << "[NSL]   " << ast.structs.size()          << " structs, "
                      << ast.functions.size()       << " functions, "
                      << ast.globals.size()         << " globals, "
                      << ast.aliases.size()         << " aliases\n";
        }

        // ==================== 7. SEMANTIC ANALYSIS ====================
        if (verbose) std::cout << "[NSL] Phase 7: Semantic analysis...\n";

        NSL::SemanticAnalyzer sema;
        if (!sema.Analyze(ast, parsedProps, diag))
        {
            diag.PrintAll();
            return 2;
        }

        diag.PrintAll();
        if (verbose && diag.ErrorCount() == 0)
            std::cout << "[NSL]   Semantic analysis passed.\n";

        // ==================== 8. ALIGNER ====================
        if (verbose) std::cout << "[NSL] Phase 8: Layout alignment...\n";

        std::unique_ptr<NSL::IAligner> aligner;
        if (targetAPI == "hlsl11" || targetAPI == "hlsl12") aligner = std::make_unique<NSL::HLSLAligner>();

        std::unordered_map<std::string, NSL::AlignedBuffer> alignedBuffers;
        std::unordered_map<std::string, NSL::AlignedStruct>  alignedStructs;

        for (const auto& group : parsedProps.groups)
        {
            if (group.resourceKind == NSL::ResourceKind::BUFFER)
            {
                auto ab = aligner->AlignBuffer(group, ast.structs);
                if (verbose)
                    std::cout << "[NSL]   Buffer '" << group.qualifiedName
                              << "' → " << ab.totalSize << " bytes\n";
                alignedBuffers[group.qualifiedName] = std::move(ab);
            }
            else if (group.resourceKind == NSL::ResourceKind::STRUCTURED_BUFFER ||
                     group.resourceKind == NSL::ResourceKind::RW_STRUCTURED_BUFFER)
            {
                auto as = aligner->AlignStructuredBuffer(group, ast.structs);
                if (verbose)
                    std::cout << "[NSL]   StructuredBuffer '" << group.qualifiedName
                              << "' → stride " << as.stride << " bytes\n";
                alignedStructs[group.qualifiedName] = std::move(as);
            }
        }

        // Sanity: every Buffer / StructuredBuffer entry must produce a resolved layout field.
        for (const auto& group : parsedProps.groups)
        {
            if (group.entries.empty()) continue;

            if (group.resourceKind == NSL::ResourceKind::BUFFER)
            {
                auto it = alignedBuffers.find(group.qualifiedName);
                if (it == alignedBuffers.end() || it->second.fields.size() != group.entries.size())
                {
                    for (const auto& entry : group.entries)
                    {
                        bool resolved = false;
                        if (it != alignedBuffers.end())
                        {
                            for (const auto& af : it->second.fields)
                            {
                                if (af.name == entry.name) { resolved = true; break; }
                            }
                        }
                        if (!resolved)
                        {
                            diag.ReportError(entry.location,
                                "Unknown or invalid type '" + entry.typeName + "' for Buffer entry '" +
                                entry.name + "' in '" + group.qualifiedName + "'.");
                        }
                    }
                    if (!diag.HasErrors())
                    {
                        diag.ReportError(group.location,
                            "Failed to compute Buffer layout for '" + group.qualifiedName
                            + "'. Check types of Add()/Get() entries.");
                    }
                }
            }
            else if (group.resourceKind == NSL::ResourceKind::STRUCTURED_BUFFER ||
                     group.resourceKind == NSL::ResourceKind::RW_STRUCTURED_BUFFER)
            {
                auto it = alignedStructs.find(group.qualifiedName);
                if (it == alignedStructs.end() || it->second.fields.size() != group.entries.size())
                {
                    for (const auto& entry : group.entries)
                    {
                        bool resolved = false;
                        if (it != alignedStructs.end())
                        {
                            for (const auto& af : it->second.fields)
                            {
                                if (af.name == entry.name) { resolved = true; break; }
                            }
                        }
                        if (!resolved)
                        {
                            diag.ReportError(entry.location,
                                "Unknown or invalid type '" + entry.typeName + "' for StructuredBuffer entry '" +
                                entry.name + "' in '" + group.qualifiedName + "'.");
                        }
                    }
                    if (!diag.HasErrors())
                    {
                        diag.ReportError(group.location,
                            "Failed to compute StructuredBuffer layout for '" + group.qualifiedName
                            + "'. Check types of Add()/Get() entries.");
                    }
                }
            }
        }
        if (diag.HasErrors())
        {
            diag.PrintAll();
            return 2;
        }

        // Also align user-defined structs from AST (used in typed StructuredBuffers)
        for (const auto& sd : ast.structs)
        {
            auto as = aligner->AlignStruct(sd, ast.structs);
            if (verbose)
                std::cout << "[NSL]   Struct '" << sd.name
                          << "' → stride " << as.stride << " bytes\n";
            alignedStructs[sd.name] = std::move(as);
        }

        // ==================== 9. HLSL CODE GENERATION ====================
        if (verbose) std::cout << "[NSL] Phase 9: HLSL code generation...\n";

        std::unique_ptr<NSL::IBackend> backend;
        if (targetAPI == "hlsl11" || targetAPI == "hlsl12") backend = std::make_unique<NSL::HLSLBackend>(targetApiEnum);

        std::string sourceCode = backend->Generate(ast, parsedProps, alignedBuffers);

        if (verbose)
            std::cout << "[NSL]   Generated " << sourceCode.size() << " characters of HLSL.\n";

        // ==================== 10. NSC WRITER ====================
        if (verbose) std::cout << "[NSL] Phase 10: Writing .nsc binary...\n";

        // Fill opaque target-API semantics for .nsc reflection (HLSL today).
        // AST / StageIOEntry keep NSL names until this boundary.
        {
            auto resolveForNSC = [&](std::vector<NSL::StageIOEntry>& entries,
                                    NSL::StageIODirection dir) -> bool
            {
                for (auto& e : entries)
                {
                    const char* sem = NSL::FindHLSLStageIOSemantic(
                        e.name.c_str(), dir, layout.stage);
                    if (!sem)
                    {
                        diag.ReportError(e.location,
                            "No target semantic mapping for stage I/O '" + e.name + "'.");
                        return false;
                    }
                    e.semantic = e.name; // Use frontend agnostic name for NSC
                }
                return true;
            };
            if (!resolveForNSC(parsedStageIO.inputs, NSL::StageIODirection::INPUT) ||
                !resolveForNSC(parsedStageIO.outputs, NSL::StageIODirection::OUTPUT))
            {
                diag.PrintAll();
                return 2;
            }
        }

        NSL::TargetAPI apiEnum = targetApiEnum;
        const uint16_t coverageFlags = CoverageCallGraphAnalyzer::ComputeFlags(ast);

        uint32_t numThreadsX = 0;
        uint32_t numThreadsY = 0;
        uint32_t numThreadsZ = 0;
        if (layout.stage == NSL::ShaderStage::COMPUTE)
        {
            auto resolveDimToken = [&](std::string tok, uint32_t& outVal) -> bool
            {
                while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.front())))
                    tok.erase(0, 1);
                while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.back())))
                    tok.pop_back();
                if (!tok.empty() && (tok.back() == 'u' || tok.back() == 'U'))
                    tok.pop_back();
                if (tok.empty()) return false;

                bool allDigits = true;
                for (char ch : tok)
                {
                    if (ch < '0' || ch > '9') { allDigits = false; break; }
                }
                if (allDigits)
                {
                    try
                    {
                        uint64_t v = std::stoull(tok);
                        if (v == 0 || v > UINT32_MAX) return false;
                        outVal = static_cast<uint32_t>(v);
                        return true;
                    }
                    catch (...) { return false; }
                }

                // Resolve against global const integer declarations
                for (const auto& g : ast.globals)
                {
                    if (g.name == tok && g.initializer)
                    {
                        if (const auto* lit = dynamic_cast<const NSL::LiteralExpr*>(g.initializer.get()))
                        {
                            if (lit->litKind == NSL::LiteralExpr::Kind::UInt &&
                                lit->uintValue > 0 &&
                                lit->uintValue <= UINT32_MAX)
                            {
                                outVal = static_cast<uint32_t>(lit->uintValue);
                                return true;
                            }
                            if (lit->litKind == NSL::LiteralExpr::Kind::Int &&
                                lit->intValue > 0 &&
                                static_cast<uint64_t>(lit->intValue) <= UINT32_MAX)
                            {
                                outVal = static_cast<uint32_t>(lit->intValue);
                                return true;
                            }
                        }
                    }
                }
                return false;
            };

            for (const auto& fn : ast.functions)
            {
                if (fn.name != "Main") continue;
                for (const auto& attr : fn.attributes)
                {
                    if (attr.name != "numthreads") continue;
                    const size_t open = attr.args.find('(');
                    const size_t close = attr.args.rfind(')');
                    if (open == std::string::npos || close == std::string::npos || close <= open + 1)
                        break;
                    std::string body = attr.args.substr(open + 1, close - open - 1);
                    const size_t c1 = body.find(',');
                    const size_t c2 = (c1 != std::string::npos) ? body.find(',', c1 + 1) : std::string::npos;
                    if (c1 != std::string::npos && c2 != std::string::npos && body.find(',', c2 + 1) == std::string::npos)
                    {
                        uint32_t tx = 0, ty = 0, tz = 0;
                        if (resolveDimToken(body.substr(0, c1), tx) &&
                            resolveDimToken(body.substr(c1 + 1, c2 - c1 - 1), ty) &&
                            resolveDimToken(body.substr(c2 + 1), tz))
                        {
                            numThreadsX = tx;
                            numThreadsY = ty;
                            numThreadsZ = tz;
                        }
                    }
                }
            }
        }

        NSL::NSCWriter writer;
        bool ok = writer.Write(
            outputFile,
            layout.stage,
            apiEnum,
            parsedProps,
            parsedStageIO,
            alignedBuffers,
            alignedStructs,
            sourceCode,
            coverageFlags,
            numThreadsX,
            numThreadsY,
            numThreadsZ
        );

        if (!ok)
        {
            std::cerr << "error: Failed to write output file '" << outputFile << "'\n";
            return 1;
        }

        std::cout << "Successfully compiled '" << inputFile << "' -> '" << outputFile
                  << "' (" << NSL::ShaderStageToString(layout.stage)
                  << ", " << parsedProps.groups.size() << " group(s))\n";

        // ==================== OPTIONAL TEXT OUTPUT ====================
        if (!textOutput.empty())
        {
            try
            {
                std::filesystem::path tp(textOutput);
                if (tp.has_parent_path())
                    std::filesystem::create_directories(tp.parent_path());
            }
            catch (...) {}

            std::ofstream textFile(textOutput);
            if (textFile.is_open())
            {
                textFile << "// NSL Compiler v" << kVersion << " — Text Debug Output\n";
                textFile << "#shader " << NSL::ShaderStageToString(layout.stage) << "\n\n";

                textFile << "// ===== PROPERTY GROUPS =====\n";
                for (const auto& g : parsedProps.groups)
                {
                    textFile << "// " << NSL::ResourceKindToString(g.resourceKind)
                             << " " << g.name
                             << " : set " << g.assignedSet << " binding " << g.assignedBinding << "\n";
                    for (const auto& e : g.entries)
                    {
                        textFile << "//   " << (e.kind == NSL::PropertyEntryKind::ADD ? "Add" : "Get")
                                 << "(" << e.name << ")";
                        if (!e.typeName.empty()) textFile << " : " << e.typeName;
                        textFile << "\n";
                    }
                }

                if (!parsedStageIO.inputs.empty() || !parsedStageIO.outputs.empty())
                {
                    textFile << "\n// ===== STAGE I/O =====\n";
                    textFile << "// INPUT_LAYOUT (" << parsedStageIO.inputs.size() << ")\n";
                    for (const auto& e : parsedStageIO.inputs)
                        textFile << "//   Get(" << e.name << ") : " << e.semantic << "\n";
                    textFile << "// OUTPUT_LAYOUT (" << parsedStageIO.outputs.size() << ")\n";
                    for (const auto& e : parsedStageIO.outputs)
                        textFile << "//   Add(" << e.name << ") : " << e.semantic << "\n";
                }

                textFile << "\n// ===== HLSL SOURCE =====\n";
                textFile << sourceCode;

                std::cout << "Text output written to '" << textOutput << "'\n";
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
