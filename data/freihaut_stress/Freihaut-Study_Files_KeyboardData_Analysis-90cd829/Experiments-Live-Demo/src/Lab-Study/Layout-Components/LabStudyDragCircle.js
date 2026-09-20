// Circle component in the Dra-and-Drop Task

import React from 'react';


export default class LabStudyDragCircle extends React.Component {

    constructor(props) {
        super(props);

        this.getMousePosition = this.getMousePosition.bind(this);
        this.startDrag = this.startDrag.bind(this);
        this.drag = this.drag.bind(this);
        this.endDrag = this.endDrag.bind(this);
    }

    // add draggin event listener
    componentDidMount() {

        this.svg.addEventListener('mousedown', this.startDrag);

        this.selectedElement = false;
        this.offset = null;

    }

    // remove dragging event listeners
    componentWillUnmount(){
        this.svg.removeEventListener('mousedown', this.startDrag);
        window.removeEventListener('mousemove', this.drag);
        window.removeEventListener('mouseup', this.endDrag);

    }

    // get the mouse position info
    getMousePosition(evt) {
        let CTM =  this.svg.getScreenCTM();
        return {
            x: (evt.clientX - CTM.e) / CTM.a,
            y: (evt.clientY - CTM.f) / CTM.d
        };
    }

    // set the position of the drag circle equal to the mouse position to achieve dragging behavior
    startDrag(evt) {
        this.selectedElement = evt.target;
        this.offset = this.getMousePosition(evt);
        this.offset.x -= parseFloat(this.selectedElement.getAttributeNS(null, "cx"));
        this.offset.y -= parseFloat(this.selectedElement.getAttributeNS(null, "cy"));

        window.addEventListener('mousemove', this.drag);
        window.addEventListener('mouseup', this.endDrag);

        // Add info that object is dragged

        this.props.dragging(true);

    }

    drag(evt) {
        if (this.selectedElement) {
            evt.preventDefault();
            let coord = this.getMousePosition(evt);
            // if the circle is dragged outside of the svg, stop it
            // || circleX >= 880 || circleY <=15 || circleY >= 530
            if (coord.x <= -15) {
                //this.selectedElement.setAttributeNS(null, "cx", 15);
                this.endDrag();
            } else if (coord.x >= 1115) {
                //this.selectedElement.setAttributeNS(null, "cx", 1080);
                this.endDrag();
            } else if (coord.y <= -15) {
                //this.selectedElement.setAttributeNS(null, "cy", 15);
                this.endDrag();
            } else if (coord.y >= 645) {
                //this.selectedElement.setAttributeNS(null, "cy", 610);
                this.endDrag();
            } else {
                this.selectedElement.setAttributeNS(null, "cx", coord.x - this.offset.x);
                this.selectedElement.setAttributeNS(null, "cy", coord.y - this.offset.y);
            }
        }
    }

    endDrag(evt) {
        // If dragging stops, return the position of the dragged circle
        this.props.returnPositions(this.selectedElement.getAttributeNS(null, "cx"), this.selectedElement.getAttributeNS(null, "cy"));

        this.selectedElement = null;

        window.removeEventListener('mousemove', this.drag);
        window.removeEventListener('mouseup', this.endDrag);

        // Add info that dragging is stopped
        this.props.dragging(false);
    }

    render() {
        return <circle cx={this.props.x} cy={this.props.y} fill={this.props.fill} r="15" ref={ref => this.svg = ref}/>
    }
}
