(function() {
var separators = [ " – ", " -- " ];

function findSeparator(container)
{
    var walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    var node;
    while ((node = walker.nextNode())) {
        for (var i = 0; i < separators.length; i++) {
            var idx = node.nodeValue.indexOf(separators[i]);
            if (idx !== -1)
                return {node : node, offset : idx, sep : separators[i]};
        }
    }
    return null;
}

// Splits <p> at the separator; wraps everything after it in .param-desc.
function reformatP(p)
{
    var found = findSeparator(p);
    if (!found)
        return null;

    var afterNode = found.node.splitText(found.offset);
    afterNode.nodeValue = afterNode.nodeValue.slice(found.sep.length);

    var range = document.createRange();
    range.setStartBefore(afterNode);
    range.setEndAfter(p.lastChild);

    var desc = document.createElement("span");
    desc.className = "param-desc";
    desc.appendChild(range.extractContents());
    p.appendChild(desc);
    return desc;
}

// Browsers auto-close nested <p> tags, so multi-paragraph descriptions
// arrive as sibling <p> elements inside <li>. Fold them into the same desc.
function reformatEntry(container)
{
    var firstP = container.querySelector(":scope > p");
    if (!firstP)
        return;

    var desc = reformatP(firstP);
    if (!desc)
        return;

    var sibling;
    while ((sibling = firstP.nextElementSibling) && sibling.tagName === "P") {
        if (desc.textContent.trim())
            desc.appendChild(document.createElement("br"));
        while (sibling.firstChild)
            desc.appendChild(sibling.firstChild);
        sibling.remove();
    }
}

document.querySelectorAll("dl.field-list dd ul.simple li")
    .forEach(reformatEntry);
document.querySelectorAll("dl.field-list dd").forEach(function(dd) {
    if (!dd.querySelector("ul.simple"))
        reformatEntry(dd);
});
}());
