(function() {
var separator = " – ";

function reformatEntry(container)
{
    var paragraphs = container.querySelectorAll(":scope > p");
    if (!paragraphs.length)
        return;

    var firstP = paragraphs[0];
    var idx = firstP.innerHTML.indexOf(separator);
    if (idx === -1)
        return;

    var before = firstP.innerHTML.substring(0, idx);
    var after = firstP.innerHTML.substring(idx + separator.length);

    var extra = [];
    for (var i = 1; i < paragraphs.length; i++) {
        extra.push(paragraphs[i].innerHTML);
        paragraphs[i].remove();
    }
    if (extra.length)
        after += (after ? "<br>" : "") + extra.join("<br>");

    firstP.innerHTML =
        before + '<br><span class="param-desc">' + after + "</span>";
}

document.querySelectorAll("dl.field-list dd ul.simple li")
    .forEach(reformatEntry);
document.querySelectorAll("dl.field-list dd").forEach(function(dd) {
    if (!dd.querySelector("ul.simple"))
        reformatEntry(dd);
});
})();
