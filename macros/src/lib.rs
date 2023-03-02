use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::{parse_macro_input, Expr, Path, Token, LitInt};

struct DimensionExpansion {
    function_path: Path,
    max_dimension: usize,
    line_points: Expr,
    cloud_points: Expr,
}

impl Parse for DimensionExpansion {
    fn parse(input: ParseStream) -> Result<Self> {
        let function_path: Path = input.parse()?;
        //dbg!(&function_path);

        let inner_parens;
        syn::parenthesized!(inner_parens in input);

        let line_points: Expr = inner_parens.parse()?;
        inner_parens.parse::<Token![,]>()?;
        let cloud_points: Expr = inner_parens.parse()?;

        input.parse::<Token![,]>()?;
        let max_dimension_literal: LitInt = input.parse()?;
        let max_dimension = max_dimension_literal.base10_digits().parse().unwrap();

        let out = DimensionExpansion {
            function_path,
            max_dimension,
            line_points,
            cloud_points,
        };

        Ok(out)
    }
}

#[proc_macro]
pub fn dimension_expansion(input: TokenStream) -> TokenStream {
    let DimensionExpansion {
        function_path,
        max_dimension,
        line_points,
        cloud_points,
    } = parse_macro_input!(input as DimensionExpansion);

    let array_dim_stream = quote!(#line_points.dim().1).into();
    let array_dim: Expr = parse_macro_input!(array_dim_stream as Expr);

    let if_kw = quote!(if);
    let else_if_kw = quote!(else if);

    let mut output = quote!();

    for i in 1..=max_dimension {

        let stmt = if i == 1 {
            if_statement(&if_kw, i, &array_dim, &function_path, &line_points, &cloud_points)
        }
        else {
            if_statement(&else_if_kw, i, &array_dim, &function_path, &line_points, &cloud_points)
        };

        output = quote!(
            #output
            #stmt
        );
    }

    output = quote!(
        #output
        else {
            panic!("unhandled dimension is greater than the maximum {}", #max_dimension);
        }
    );


    TokenStream::from(output)
}

fn if_statement(
    if_keyword: &TokenStream2,
    target_dimension: usize,
    actual_dimension: &Expr,
    fn_path: &Path,
    line_points: &Expr,
    cloud_points: &Expr,
) -> TokenStream2 {
    quote!(
        #if_keyword #actual_dimension == #target_dimension {
            #fn_path::<#target_dimension>(#line_points, #cloud_points)
        }
    )
}
