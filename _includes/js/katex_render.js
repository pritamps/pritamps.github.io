
$("script[type='math/tex']").replaceWith(
  function(){
    var tex = $(this).text();
    return "<span class=\"inline-equation\">" + 
           katex.renderToString(tex.replace("&","")) +
           "</span>";
});

$("script[type='math/tex; mode=display']").replaceWith(
  function(){
    var tex = $(this).text();

    // Need to do this replace here because Javascript 
    // doesn't treat the `&` properly, and the `&` is needed
    // for alignment
    tex = tex.replace(/&amp;/gi, "\u0026")
    return "<div class=\"equation\" align=\"center\">" + 
           katex.renderToString(tex) +
           "</div> <p></p>";
});
