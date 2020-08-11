"""Events relating to the mouse"""
from OpenGLContext.events import event, eventmanager
from OpenGL.GLU import *
from OpenGLContext.scenegraph import nodepath
from OpenGLContext.arrays import array

class MouseEvent( event.Event ):
    """Base class for all mouse-based events.

    Provides a number of utility methods for
    interacting with the mouse event and the
    select render mode.

    Attributes:
        type -- Event type string specifier
        viewCoordinate -- 3D vector in view-space of
            nearest-geometry intersection
        worldCoordinate -- 3D vector in model-space of
            nearest-geometry intersection
        pickPoint -- 2D view-space coordinate for mouse
        objectPaths -- NodePaths from scenegraph to each
            shape under the cursor, arranged in nearest-
            to-farthest order
        stopPropagation -- flag which, when set, stops
            the event from continuing along the capture/
            bubbling ProcessEvent path.  See:
                eventhandler.BubblingEventManager
        processMorePaths -- if true (false by default),
            then continue processing node-paths after the
            current node-path is complete.
        
    (Low-level data attributes):
        nameStack -- low-level OpenGL name-stack with
            GL names and distances:
            [ (near,far,[name,...]),...]
        modelViewMatrix -- active model-view matrix for
            the context on this rendering pass
        projectionMatrix -- active projection matrix for
            the context on this rendering pass
        viewport -- viewport dimensions
    """
    type = ""
    viewCoordinate = ()
    pickPoint = ()
    worldCoordinate = ()
    nameStack = ()
    modelViewMatrix = None
    projectionMatrix = None
    viewport = None
    objectPaths = None
    stopPropagation = 0
    processMorePaths = 0
    currentPath = ()
    currentNode = None
    atTarget = 0
    def getPickPoint( self ):
        """Get the 2D picking point in OpenGL coordinates
        
        Note that OpenGL coordinates are counted from the bottom left
        corner of the window, in contrast to most windowing libraries
        which use the upper left corner as the origin.
        """
        return self.pickPoint
    def getNameStack( self ):
        """Get the name-stack as reported by the rendering pass."""
        return self.nameStack
    def setNameStack( self, stack ):
        """Set the name-stack.  Called by the select render mode"""
        self.nameStack = stack
    def setObjectPaths( self, paths ):
        """Set the object-path sets"""
        self.objectPaths = paths
    def getObjectPaths( self ):
        """Get the object-path object (None if not set)"""
        return self.objectPaths
    def unproject(self, viewCoordinate=None):
        """Get the world coordinates for viewCoordinate for the event

        viewCoordinate -- coordinate to project, if omitted, the
            first "hit" entry in the name-stack will be used.  Otherwise
            should be a three-item array/tuple.  The z component of
            this tuple is interpreted acording to type.  If it's an
            integer or long, it's consider a raw OpenGL depth-buffer
            value as returned in the name-stack, and is converted to
            a floating-point depth value.  If it's already a floating-
            point value, it's left as-is.
        """
        if viewCoordinate is None:
            if self.worldCoordinate:
                return self.worldCoordinate
            if not self.viewCoordinate:
                if self.getNameStack():
                    x,y = self.getPickPoint()
                    z = self.getNameStack()[0][0] # near of the first result-set
                    z = z / (2.0**32-1.0)
                    self.viewCoordinate = x,y,z
                else:
                    raise ValueError( """Attempting to unproject a mouse event without a viewCoordinate or namestack %s"""%self)
            else:
                x,y,z = self.viewCoordinate
        else:
            x,y,z = viewCoordinate
            if isinstance( z, (int,long)):
                z = z / (2.0**32-1.0)
        viewport = self.viewport
        if viewport is not None:
            viewport = array(viewport,'i')
        worldCoordinate = gluUnProject(
            x,y,z,
            self.modelViewMatrix.astype('d'),
            self.projectionMatrix.astype('d'),
            viewport,
        )
        if viewCoordinate is None:
            self.worldCoordinate = worldCoordinate
        return worldCoordinate
        
    def project (self, worldCoordinate=None):
        """Get the screen coordinates for the event w/out
        the pick-point requires that worldCoordinate and
        rendering pass be available or that the
        viewCoordinate field already be available.
        """
        if worldCoordinate is None:
            if self.viewCoordinate:
                return self.viewCoordinate
            x,y,z = self.worldCoordinate
        else:
            x,y,z = worldCoordinate
        viewCoordinate = gluProject(
            x,y,z,
            self.modelViewMatrix,
            self.projectionMatrix,
            self.viewport,
        )
        if worldCoordinate is None:
            self.viewCoordinate = viewCoordinate
        return viewCoordinate


class MouseButtonEvent (MouseEvent):
    """Mouse event representing a change of state for a mouse button
    attributes:
        type -- "mousebutton"
        renderingPass -- pointer to the OpenGLContext.renderpass.RenderPass
            object associated with this event
        modifiers -- three-tuple of booleans: (shift, control, alt)
        button -- the "button ID" which changed state
            Valid Values:
                0 -- first mouse button, "left"
                1 -- second mouse button, "right"
                2 -- third mouse button, "middle"
                n -- nth mouse button, device-specific
        state -- Boolean 0 = up/released, 1 = down/pressed
        nameStack -- selection name stack, [(near, far, [names,...]),...]
    """
    type = "mousebutton"
    button = -1 # which button was depressed
    state = 0 # the new state of the button, 0 or 1
    def getKey (self):
        """Get the event key used to lookup a handler for this event"""
        return (self.button, self.state, self.getModifiers(),)

class MouseEventManager( eventmanager.BubblingEventManager ):
    """Manager base-class for mouse-related events"""

class MouseButtonEventManager (MouseEventManager):
    """Manager for MouseButtonEvent instances"""
    type = MouseButtonEvent.type
    def registerCallback(
        self,
        button= 0, state=0, modifiers = (0,0,0),
        function = None, node = None, capture = 0,
    ):
        """Register a function to receive mouse-button events matching
        the given specification  To deregister, pass None as the
        function.

        button -- integer name for the button in which you are interested
            Valid Values:
                0 -- first mouse button, "left"
                1 -- second mouse button, "right"
                2 -- third mouse button, "middle"
                n -- nth mouse button, device-specific
        state -- integer state values
            Value Values:
                0/1 -- for all boolean buttons, 0 == up/released, 1 = down/pressed
            
        modifiers -- (shift, control, alt) as a tuple of booleans.
        function -- function taking a single argument (a KeypressEvent)
            or None to deregister the callback.
        node -- if non-None, a node/object which limits scope of
            the binding.  Only the node, or children of the node
            can generate events.  Otherwise watches for "anonymous"
            messages, those which are not handled by anything else.
        capture -- if true, function is called before the target's
            bindings, allowing for parents to override children's
            bindings.
            
        returns the previous handler or None
        """
        if button is not None:
            key = button, state, modifiers
        else:
            key = None
        return super( MouseButtonEventManager, self).registerCallback(
            key, function, node, capture,
        )

class MouseMoveEvent( MouseEvent ):
    """Event representing a change of position for the mouse.

    Can represent either a "free move" or a "drag" with buttons
    depressed.  See the buttons field for a list of the button IDs
    active while moving

    attributes:
        type -- "mousemove"
        renderingPass -- pointer to the OpenGLContext.renderpass.RenderPass
            object associated with this event
        modifiers -- three-tuple of booleans: (shift, control, alt)
        buttons -- tuple of active buttons (in ascending order)
            Valid Values:
                () -- move with no buttons
                (n,) -- drag with single button 'n' depressed
                (n,n+x) -- drag with two buttons depressed
        nameStack -- selection name stack, [(near, far, [names,...]),...]
    """
    type = "mousemove"
    dragStart = () # if non-null, the initial position of the drag (viewCoordinates)
    buttons = ()
    def getKey (self):
        """Get the event key used to lookup a handler for this event"""
        return self.getButtons(), self.getModifiers()
    def getButtons( self ):
        """Return the active buttons as a tuple of integers."""
        return self.buttons

class MouseMoveEventManager (MouseEventManager):
    """Manager for MouseMoveEvent instances

    lastPath -- tracks the last-pointed-to path, used to
        generate mousein and mouseout event types.
    """
    type = MouseMoveEvent.type
    lastPath = ()
    def registerCallback(
        self,
        buttons = (), modifiers = (0,0,0),
        function = None, node=None, capture=0
    ):
        """Register a function to receive keyboard events matching
        the given specification  To deregister, pass None as the
        function.

        buttons -- tuple of active buttons (in ascending order)
            Valid Values:
                () -- move with no buttons
                (n,) -- drag with single button 'n' depressed
                (n,n+x) -- drag with two buttons depressed
        modifiers -- (shift, control, alt) as a tuple of booleans.
        function -- function taking a single argument (a KeypressEvent)
            or None to deregister the callback.
            
        returns the previous handler or None
        """
        if buttons:
            key = buttons, modifiers
        else:
            key = None
        return super( MouseMoveEventManager, self).registerCallback(
            key, function, node, capture,
        )
    def ProcessEvent(self, event):
        """Dispatch an incoming event

        This method tracks previously-pointed paths and generates
        synthetic "mousein" and "mouseout" events
        """
        if event.getObjectPaths():
            newPath = event.getObjectPaths()[0]
        else:
            newPath = ()
        # have we moved away from a last-known path?
        lastPath = self.lastPath
        if (lastPath and lastPath != newPath):
            # we were previously pointing at something...
            # create a new event "mouseout" and tell context to process it...
            event.context.ProcessEvent(
                MouseOutEvent.fromMoveEvent( event, lastPath, newPath)
            )
        if newPath and newPath != lastPath:
            # we are pointing at a new object...
            event.context.ProcessEvent(
                MouseInEvent.fromMoveEvent( event, lastPath, newPath)
            )
        result = super( MouseMoveEventManager, self).ProcessEvent( event )
        self.lastPath = newPath
        return result
            
class _MouseChangeEvent( MouseEvent ):
    """Base class for mouse in/out events

    The mouse in/out event types are "synthetic",
    that is, they are generated by other events
    when certain conditions are true.  The change
    events allow for easily constructing common
    interface elements such as mouse-overs.

    Attributes:
        lastPath -- previous value for target path,
            in essence, the state from which we have
            come
        newPath -- new value for target path,
            in essence, the new state to which we have
            just changed

    The change event also includes all attributes
    of the MouseMoveEvent which triggered the change,
    see the MouseMoveEvent class for details.

    Change events are sent anonymously and from
    the set of nodes which have _changed_ in the path,
    rather than all nodes in the new/old path.
    """
    lastPath = ()
    newPath = ()
    def __init__( self, **named ):
        """Initialise the event with named attributes"""
        for key,value in named.items():
            setattr( self, key, value )
        super( _MouseChangeEvent, self).__init__()
    def fromMoveEvent( cls, event, lastPath, newPath ):
        """Construct synthetic mouse event from a move event"""
        base = event.__dict__.copy()
        try:
            del base['visitedNodes']
        except KeyError:
            pass
        return cls( lastPath=lastPath, newPath=newPath, **base )
    fromMoveEvent = classmethod( fromMoveEvent )
class MouseInEvent( _MouseChangeEvent ):
    """Mouse has just begun pointing to a particular path

    The MouseInEvent indicates that the pointer has just
    begun pointing to the geometry at the end of the
    newPath attribute.

    * only sent when the newPath is non-null (i.e. there
        is actual geometry under the pointer).
    * sent after any corresponding MouseOutEvent for the
        lastPath has been sent
    """
    type = "mousein"
class MouseOutEvent( _MouseChangeEvent ):
    """Mouse has just stopped pointing to a particular path

    The MouseOutEvent indicates that the pointer has just
    stopped pointing to the geometry at the end of the
    lastPath attribute.

    * only sent when the lastPath is non-null (i.e. there
        was actual geometry under the pointer).
    * sent before any corresponding MouseInEvent for the
        newPath has been sent
    """
    type = "mouseout"
    
class _MouseChangeEventManager (MouseEventManager):
    """Manager for _MouseChangeEvent instances
    """
    def registerCallback(
        cls,
        buttons = (), modifiers = (0,0,0),
        function = None, node=None, capture=0
    ):
        """Register a function to receive keyboard events matching
        the given specification  To deregister, pass None as the
        function.

        buttons -- tuple of active buttons (in ascending order)
            Valid Values:
                () -- move with no buttons
                (n,) -- drag with single button 'n' depressed
                (n,n+x) -- drag with two buttons depressed
        modifiers -- (shift, control, alt) as a tuple of booleans.
        function -- function taking a single argument (a KeypressEvent)
            or None to deregister the callback.
            
        returns the previous handler or None
        """
        if buttons:
            key = buttons, modifiers
        else:
            key = None
        return super( _MouseChangeEventManager, cls).registerCallback(
            key, function, node, capture,
        )
    registerCallback = classmethod( registerCallback )
class MouseInEventManager( _MouseChangeEventManager ):
    type = 'mousein'
    def _traversalPaths( self, event ):
        """Get the paths to traverse for a given event

        In is done on the delta between the previous
        and the new items, specifically those items which
        are now under the mouse which were not previously.
        """
        shared = event.newPath.common( event.lastPath )
        return (event.newPath[len(shared):], )
class MouseOutEventManager( _MouseChangeEventManager):
    type = 'mouseout'
    def _traversalPaths( self, event ):
        """Get the paths to traverse for a given event

        Out is done on the delta between the previous
        and the new items, specifically those items which
        were previously under the mouse, but no longer are.
        """
        shared = event.lastPath.common( event.newPath )
        return (event.lastPath[len(shared):], )