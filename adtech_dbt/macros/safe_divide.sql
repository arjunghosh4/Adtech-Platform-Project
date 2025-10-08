{% macro safe_divide(numerator, denominator, null_result=0) %}
    (case
        when {{ denominator }} is null or {{ denominator }} = 0 then {{ null_result }}
        else {{ numerator }}::numeric / {{ denominator }}::numeric
     end)
{% endmacro %}