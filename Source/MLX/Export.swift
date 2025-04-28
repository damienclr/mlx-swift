// Copyright © 2025 Apple Inc.

import Cmlx
import Foundation

public func exportFunction(
    to url: URL, shapeless: Bool = false, _ f: @escaping ([MLXArray]) -> [MLXArray]
) -> FunctionExporterSingle {
    FunctionExporterSingle(url: url, shapeless: shapeless, f: f)
}

public func exportFunctions(
    to url: URL, shapeless: Bool = false, _ f: @escaping ([MLXArray]) -> [MLXArray],
    export: (FunctionExporterMultiple) throws -> Void
) throws {
    let exporter = try FunctionExporterMultiple(url: url, shapeless: shapeless, f: f)
    try export(exporter)
}

@dynamicCallable
public final class FunctionExporterSingle {
    let url: URL
    let shapeless: Bool
    let f: ([MLXArray]) -> [MLXArray]

    internal init(url: URL, shapeless: Bool, f: @escaping ([MLXArray]) -> [MLXArray]) {
        self.url = url
        self.shapeless = shapeless
        self.f = f
    }

    public func dynamicallyCall(withKeywordArguments args: KeyValuePairs<String, MLXArray>) throws {
        let positionalArgs = mlx_vector_array_new()
        defer { mlx_vector_array_free(positionalArgs) }
        for (key, value) in args {
            if key.isEmpty {
                mlx_vector_array_append_value(positionalArgs, value.ctx)
            }
        }

        let keys = args.compactMap { $0.key.isEmpty ? nil : $0.key }
        let kwargs = new_mlx_array_map(
            Dictionary(
                args.compactMap { $0.key.isEmpty ? nil : ($0.key, $0.value) },
                uniquingKeysWith: { a, b in a }))
        defer { mlx_map_string_to_array_free(kwargs) }

        let closure = new_mlx_kwargs_closure(keys: keys, f)
        defer { mlx_closure_kwargs_free(closure) }

        _ = try withError {
            mlx_export_function_kwargs(url.path, closure, positionalArgs, kwargs, shapeless)
        }
    }
}

@dynamicCallable
public final class FunctionExporterMultiple {
    let exporter: mlx_function_exporter

    internal init(url: URL, shapeless: Bool = false, f: @escaping ([MLXArray]) -> [MLXArray]) throws
    {
        let closure = new_mlx_closure(f)
        defer { mlx_closure_free(closure) }

        self.exporter = try withError {
            mlx_function_exporter_new(url.path, closure, shapeless)
        }
    }

    deinit {
        mlx_function_exporter_free(exporter)
    }

    public func dynamicallyCall(withKeywordArguments args: KeyValuePairs<String, MLXArray>) throws {
        let positionalArgs = mlx_vector_array_new()
        defer { mlx_vector_array_free(positionalArgs) }
        for (key, value) in args {
            if key.isEmpty {
                mlx_vector_array_append_value(positionalArgs, value.ctx)
            }
        }

        let kwargs = new_mlx_array_map(
            Dictionary(
                args.compactMap { $0.key.isEmpty ? nil : ($0.key, $0.value) },
                uniquingKeysWith: { a, b in a }))
        defer { mlx_map_string_to_array_free(kwargs) }

        _ = try withError {
            mlx_function_exporter_apply_kwargs(exporter, positionalArgs, kwargs)
        }
    }
}

public func importFunction(from url: URL) throws -> ImportedFunction {
    try ImportedFunction(url: url)
}

@dynamicCallable
public final class ImportedFunction {

    private let ctx: mlx_imported_function

    public init(url: URL) throws {
        self.ctx = try withError {
            mlx_imported_function_new(url.path)
        }
    }

    deinit {
        mlx_imported_function_free(ctx)
    }

    public func dynamicallyCall(withKeywordArguments args: KeyValuePairs<String, MLXArray>) throws
        -> [MLXArray]
    {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }

        let positionalArgs = mlx_vector_array_new()
        defer { mlx_vector_array_free(positionalArgs) }
        for (key, value) in args {
            if key.isEmpty {
                mlx_vector_array_append_value(positionalArgs, value.ctx)
            }
        }

        let kwargs = new_mlx_array_map(
            Dictionary(
                args.compactMap { $0.key.isEmpty ? nil : ($0.key, $0.value) },
                uniquingKeysWith: { a, b in a }))
        defer { mlx_map_string_to_array_free(kwargs) }

        _ = try withError {
            mlx_imported_function_apply_kwargs(&result, ctx, positionalArgs, kwargs)
        }

        return mlx_vector_array_values(result)
    }
}
