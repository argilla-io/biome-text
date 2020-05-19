<%
  import os

  import pdoc
  from pdoc.html_helpers import extract_toc, glimpse, to_html as _to_html, format_git_link


  def link(dobj: pdoc.Doc, name=None):
    name = name or dobj.qualname + ('()' if isinstance(dobj, pdoc.Function) else '')
    if isinstance(dobj, pdoc.External) and not external_links:
        return name
    url = dobj.url(relative_to=module, link_prefix=link_prefix,
                   top_ancestor=not show_inherited_members)
    return '<a title="{}" href="{}">{}</a>'.format(dobj.refname, url, name)


  def to_html(text):
    return _to_html(text, docformat=docformat, module=module, link=link, latex_math=latex_math)


  def get_annotation(bound_method, sep=':'):
    annot = show_type_annotations and bound_method(link=link) or ''
    if annot:
        annot = ' ' + sep + '\N{NBSP}' + annot
    return annot
%>
<%def name="ident(name)"><span class="ident">${name}</span></%def>

<%def name="show_func(f,label)">

<pre class="title">

${"### "}${f.name} <Badge text="${label}"/>
</pre>

    <dt>
      <%
          params = f.params(annotate=show_type_annotations, link=link)
          return_type = get_annotation(f.return_annotation, '->')
          params_list = True
          if len(params) <=1:
              params = ', '.join(params)
              params_list = False
      %>
      ## FUNCTION, METHODS CODE DEFINITION VIEW 
      <div class="language-python extra-class">
<pre class="language-python">
<code>
% if not params_list:
<span class="token keyword">${f.funcdef()}</span> ${ident(f.name)}</span>(<span>${params})${return_type}</span>
% else:
<span class="token keyword">${f.funcdef()}</span> ${ident(f.name)} (</span>
% for p in params:
  ${p},
% endfor
) ${return_type}
% endif
</code>
</pre>
      </div>
    </dt>
    <dd>${show_desc(f)}</dd>
  </%def>

<%def name="show_desc(d, short=False)">
  <%
  inherits = ' inherited' if d.inherits else ''
  docstring = glimpse(d.docstring) if short or inherits else d.docstring
  %>
  % if d.inherits:
          <em>Inherited from:</em>
          % if hasattr(d.inherits, 'cls'):
              <code>${link(d.inherits.cls)}</code>.<code>${link(d.inherits, d.name)}</code>
          % else:
              <code>${link(d.inherits)}</code>
          % endif
  % endif
  ${docstring | to_html}
</%def>

<%def name="show_module_list(modules)">
  <h1>Python module list</h1>

  % if not modules:
    <p>No modules found.</p>
  % else:
    <dl id="http-server-module-list">
    % for name, desc in modules:
        <div class="flex">
        <dt><a href="${link_prefix}${name}">${name}</a></dt>
        <dd>${desc | glimpse, to_html}</dd>
        </div>
    % endfor
    </dl>
  % endif
</%def>

<%def name="show_column_list(items)">
  <%
      two_column = len(items) >= 6 and all(len(i.name) < 20 for i in items)
  %>
  <ul class="${'two-column' if two_column else ''}">
  % for item in items:
    <li><code>${link(item, item.name)}</code></li>
  % endfor
  </ul>
</%def>

<%def name="show_module(module)">
  <%
  variables = module.variables(sort=sort_identifiers)
  classes = module.classes(sort=sort_identifiers)
  functions = module.functions(sort=sort_identifiers)
  submodules = module.submodules()
  %>

  # ${module.name} ${'<Badge text="Namespace"/>' if module.is_namespace else  \
'<Badge text="Package"/>' if module.is_package and not module.supermodule else \
'<Badge text="Module"/>'}
<div></div>

  ${module.docstring | to_html}
  
    % if submodules:
    <h2 class="section-title" id="header-submodules">Sub-modules</h2>
    % for m in submodules:
      <code class="name">${link(m)}</code>
      ${show_desc(m, short=True)}
    % endfor
    % endif

    % if variables:
    <h2 class="section-title" id="header-variables">Global variables</h2>
    % for v in variables:
      <% return_type = get_annotation(v.type_annotation) %>
      <dt id="${v.refname}"><code class="name">var ${ident(v.name)}${return_type}</code></dt>
      ${show_desc(v)}
    % endfor
    % endif

    % if functions:
    % for f in functions:
      ${show_func(f, "Function")}
    % endfor
    % endif
    % if classes:
    % for c in classes:
      <%
      class_vars = c.class_variables(show_inherited_members, sort=sort_identifiers)
      smethods = c.functions(show_inherited_members, sort=sort_identifiers)
      inst_vars = c.instance_variables(show_inherited_members, sort=sort_identifiers)
      methods = c.methods(show_inherited_members, sort=sort_identifiers)
      mro = c.mro()
      subclasses = c.subclasses()
      params = c.params(annotate=show_type_annotations, link=link)
      params_list = True
      if len(params) <=2:
        params = ', '.join(params)
        params_list = False
      %>

<div></div>
<pre class="title">
 
${"## "}${c.name} <Badge text="Class"/>
</pre>

      ## CLASS DEFINITION VIEW
   
   
<pre class="language-python">
  <code>
    % if params_list:
    <span class="token keyword">class</span> ${ident(c.name)} (</span>
    % for p in params:
        <span>${p}</span><span>,</span>
    % endfor
    <span>)</span>
    % else:
    <span class="token keyword">class</span> ${ident(c.name)} (${params})</span>
    % endif
  </code>
</pre>
     
     

      ${show_desc(c)}


      % if mro:

<pre class="title">


${"### "}${"Ancestors"}
</pre>


          <ul class="hlist">
          % for cls in mro:
              <li>${link(cls)}</li>
          % endfor
          </ul>
      %endif

      % if subclasses:
<pre class="title">

${"### "}${"Subclasses"}
</pre>


          <ul class="hlist">
          % for sub in subclasses:
              <li>${link(sub)}</li>
          % endfor
          </ul>
      % endif
      % if class_vars:

<pre class="title">


${"### Class variables"}
</pre>

          <dl>
          % for v in class_vars:
              <% return_type = get_annotation(v.type_annotation) %>
              <dt id="${v.refname}"><code class="name">var ${ident(v.name)}${return_type}</code></dt>
              <dd>${show_desc(v)}</dd>
          % endfor
          </dl>
      % endif
      % if smethods:
          <dl>
          % for f in smethods:
              ${show_func(f, "Static method")}
          % endfor
          </dl>
      % endif
      % if inst_vars:

<pre class="title">


${"### Instance variables"}
</pre>

          <dl>
          % for v in inst_vars:
              <% return_type = get_annotation(v.type_annotation) %>
              <dt id="${v.refname}"><code class="name">var ${ident(v.name)}${return_type}</code></dt>
              <dd>${show_desc(v)}</dd>
          % endfor
          </dl>
      % endif
      % if methods:
          <dl>
          % for f in methods:
              ${show_func(f, "Method")}
          % endfor
          </dl>
      % endif

      % if not show_inherited_members:
          <%
              members = c.inherited_members()
          %>
          % if members:

<pre class="title">


${"### Inherited members"}
</pre>

              <ul class="hlist">
              % for cls, mems in members:
                  <li><code><b>${link(cls)}</b></code>:
                      <ul class="hlist">
                          % for m in mems:
                              <li><code>${link(m, name=m.name)}</code></li>
                          % endfor
                      </ul>

                  </li>
              % endfor
              </ul>
          % endif
      % endif
    % endfor
    % endif
</%def>

<%def name="module_index(module)">
  <%
  variables = module.variables(sort=sort_identifiers)
  classes = module.classes(sort=sort_identifiers)
  functions = module.functions(sort=sort_identifiers)
  submodules = module.submodules()
  supermodule = module.supermodule
  %>
 </%def>

  % if module_list:
      ${show_module_list(modules)}
  % else:
      ${show_module(module)}
    ${module_index(module)}
  % endif

## <div class="git-link-div"><a href="${git_link}" class="git-link">Browse git</a></div> might be useful for linking with github